import torch
import torch.nn as nn

import numpy as np

np.random.seed(0)

from model.generate_anchor import generate_anchors
from model.bbox_transform import bbox_overlaps_batch, bbox_transform


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(
            batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret


class BboxTargetLayer(nn.Module):
    def __init__(self, cfg):
        super(BboxTargetLayer, self).__init__()
        self._cfg = dict(cfg)
        self._preprocess()

    def _preprocess(self):
        # pre-computing stuff for making anchor later
        allowed_border = 0
        im_info = (self._cfg['MAX_SIZE'], self._cfg['MAX_SIZE'])
        base_anchors = generate_anchors(
            base_size=self._cfg['RPN_FEAT_STRIDE'],
            ratios=self._cfg['ANCHOR_RATIOS'],
            scales=np.array(self._cfg['ANCHOR_SCALES'], dtype=np.float32))
        num_anchors = base_anchors.shape[0]
        feat_stride = self._cfg['RPN_FEAT_STRIDE']
        feat_width = self._cfg['MAX_SIZE'] // self._cfg['RPN_FEAT_STRIDE']
        feat_height = feat_width

        shift_x = np.arange(0, feat_width) * feat_stride
        shift_y = np.arange(0, feat_height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = num_anchors
        K = shifts.shape[0]
        all_anchors = base_anchors.reshape((1, A, 4)) + \
            shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -allowed_border) &
            (all_anchors[:, 1] >= -allowed_border) &
            (all_anchors[:, 2] < im_info[1] + allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + allowed_border)    # height
        )[0]

        anchors = all_anchors[inds_inside, :]

        self._A = A
        self._feat_height = feat_height
        self._feat_width = feat_width
        self._total_anchors = total_anchors
        self._inds_inside = torch.from_numpy(inds_inside).long()
        self._anchors = torch.from_numpy(anchors).float()

    def cuda(self, device=None):
        self._inds_inside = self._inds_inside.cuda(device)
        self._anchors = self._anchors.cuda(device)
        return self._apply(lambda t: t.cuda(device))

    def forward(self, gt_boxes):
        batch_size = gt_boxes.size(0)

        labels = gt_boxes.new(batch_size, self._inds_inside.size(0)).fill_(-1)

        overlaps = bbox_overlaps_batch(self._anchors, gt_boxes)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < self._cfg['TRAIN.RPN_NEGATIVE_OVERLAP']] = 0

        # mark max overlap of 0, which are padded gt_boxes
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
        # mark the max overlap of each ground truth
        keep = torch.sum(overlaps.eq(
            gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)
        if torch.sum(keep) > 0:
            labels[keep > 0] = 1

        labels[max_overlaps >= self._cfg['TRAIN.RPN_POSITIVE_OVERLAP']] = 1

        # subsample positive labels if we have too many
        num_fg = int(self._cfg['TRAIN.RPN_FG_FRACTION'] *
                     self._cfg['TRAIN.RPN_BATCHSIZE'])

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        bbox_targets = gt_boxes.new(
            batch_size, self._inds_inside.size(0), 4).fill_(0)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.from_numpy(
                    np.random.permutation(fg_inds.size(0))
                ).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            num_bg = self._cfg['TRAIN.RPN_BATCHSIZE'] - \
                torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.from_numpy(
                    np.random.permutation(bg_inds.size(0))
                ).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

            bbox_targets[i] = bbox_transform(
                self._anchors, gt_boxes[i][argmax_overlaps[i], :4])

        # map up to original set of anchors
        labels = _unmap(
            labels, self._total_anchors, self._inds_inside, batch_size,
            fill=-1)
        bbox_targets = _unmap(
            bbox_targets, self._total_anchors, self._inds_inside, batch_size,
            fill=0)

        labels = labels.view(
            batch_size, self._feat_height, self._feat_width, self._A, 1
        ).contiguous()
        bbox_targets = bbox_targets.view(
            batch_size, self._feat_height, self._feat_width, self._A, 4
        ).contiguous()

        return (labels, bbox_targets)
