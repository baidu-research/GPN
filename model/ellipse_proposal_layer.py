import torch
import torch.nn as nn

import numpy as np

np.random.seed(0)

from model.generate_anchor import generate_anchors
from model.bbox_transform import clip_boxes
from model.ellipse_transform import ellipse_transform_inv, ellipse2box
from nms.cpu_nms import cpu_nms
from nms.gpu_nms import gpu_nms


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().view(-1)

    return keep


class EllipseProposalLayer(nn.Module):
    def __init__(self, cfg):
        super(EllipseProposalLayer, self).__init__()
        self._cfg = dict(cfg)
        self._preprocess()

    def _preprocess(self):
        # pre-computing stuff for making anchor later
        self._im_info = (self._cfg['MAX_SIZE'], self._cfg['MAX_SIZE'])
        base_anchors = generate_anchors(
            base_size=self._cfg['RPN_FEAT_STRIDE'],
            ratios=[1],
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
        anchors = base_anchors.reshape((1, A, 4)) + \
            shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        self._feat_height = feat_height
        self._feat_width = feat_width
        self._anchors = torch.from_numpy(anchors).float()

    def cuda(self, device=None):
        self._anchors = self._anchors.cuda(device)
        return self._apply(lambda t: t.cuda(device))

    def forward(self, out_cls, out_ellipse):
        """
        out_cls: (feat_height, feat_width, anchors, 2) FloatVariable
        out_ellipse: (feat_height, feat_width, anchors, 5) FloatVariable
        """
        scores = nn.functional.softmax(
            out_cls, dim=3)[..., 1].contiguous().data.view(-1, 1)
        ellipse_deltas = out_ellipse.data.view(-1, 5)

        # 1. Generate proposals from ellipse deltas and shifted anchors
        # Convert anchors into proposals via ellipse transformations
        # Convert ellipse into bbox proposals
        ellipses = ellipse_transform_inv(self._anchors, ellipse_deltas)
        boxes = ellipse2box(ellipses, self._cfg['ELLIPSE_PAD'])

        # 2. clip predicted boxes to image
        boxes = clip_boxes(boxes, self._im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTICE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(boxes, self._cfg['TEST.RPN_MIN_SIZE'])
        boxes = boxes[keep, :]
        ellipses = ellipses[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        _, order = torch.sort(scores.view(-1), dim=0, descending=True)
        if self._cfg['TEST.RPN_PRE_NMS_TOP_N'] > 0:
            order = order[:self._cfg['TEST.RPN_PRE_NMS_TOP_N']]
        boxes = boxes[order, :]
        ellipses = ellipses[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if self._cfg['USE_GPU_NMS']:
            nms = gpu_nms
        else:
            nms = cpu_nms
        dets = np.hstack((boxes.cpu().numpy(), scores.cpu().numpy()))
        keep = nms(dets, self._cfg['TEST.RPN_NMS_THRESH'])
        keep = torch.from_numpy(np.array(keep)).type_as(scores).long()
        if self._cfg['TEST.RPN_POST_NMS_TOP_N'] > 0:
            keep = keep[:self._cfg['TEST.RPN_POST_NMS_TOP_N']]
        boxes = boxes[keep, :]
        ellipses = ellipses[keep, :]
        scores = scores[keep].view(-1)

        return (boxes, ellipses, scores)
