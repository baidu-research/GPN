import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from model.generate_anchor import generate_anchors


class LossCls(nn.Module):

    def __init__(self):
        super(LossCls, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, out_cls, labels):
        label_idcs = labels.view(-1).ne(-1).nonzero().view(-1)
        out_cls_keep = torch.index_select(out_cls.view(-1, 2), 0, label_idcs)
        labels_keep = torch.index_select(labels.view(-1), 0, label_idcs)

        loss = self.cls_loss(out_cls_keep,
                             labels_keep.type(torch.cuda.LongTensor))

        return loss


class LossBbox(nn.Module):

    def __init__(self):
        super(LossBbox, self).__init__()
        self.bbox_loss = nn.SmoothL1Loss()

    def forward(self, out_bbox, labels, bbox_targets):
        pos_idcs = labels.view(-1).eq(1).nonzero().view(-1)
        out_bbox_keep = torch.index_select(out_bbox.view(-1, 4), 0,
                                           pos_idcs)
        bbox_targets_keep = torch.index_select(bbox_targets.view(-1, 4), 0,
                                               pos_idcs)

        loss = self.bbox_loss(out_bbox_keep, bbox_targets_keep)

        return loss


class LossEllipseSL1(nn.Module):

    def __init__(self):
        super(LossEllipseSL1, self).__init__()
        self.ellipse_loss = nn.SmoothL1Loss()

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, out_ellipse, labels, ellipse_targets):
        pos_idcs = labels.view(-1).eq(1).nonzero().view(-1)
        out_ellipse_keep = torch.index_select(out_ellipse.view(-1, 5), 0,
                                              pos_idcs)
        ellipse_targets_keep = torch.index_select(ellipse_targets.view(-1, 5),
                                                  0, pos_idcs)

        loss = self.ellipse_loss(out_ellipse_keep, ellipse_targets_keep)

        return loss


class LossEllipseKLD(nn.Module):

    def __init__(self, cfg):
        super(LossEllipseKLD, self).__init__()
        self._cfg = dict(cfg)
        self._preprocess()

    def _preprocess(self):
        # pre-computing stuff for making anchor later
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

        # self.anchors = torch.from_numpy(anchors).float()
        # self.anchors = nn.Parameter(torch.from_numpy(anchors).float(),
        #                             requires_grad=False)
        self._anchors = Variable(torch.from_numpy(anchors).float(),
                                 requires_grad=False)

    def cuda(self, device=None):
        self._anchors = self._anchors.cuda(device)
        return self._apply(lambda t: t.cuda(device))

    def forward(self, out_ellipse, labels, ellipse_targets):
        batch_size = out_ellipse.size(0)
        anchors = self._anchors.repeat(batch_size, 1)
        # anchors = Variable(
        #     self.anchors.repeat(batch_size, 1).type_as(out_ellipse.data),
        #     requires_grad=False)

        pos_idcs = labels.view(-1).eq(1).nonzero().view(-1)
        out_ellipse_keep = torch.index_select(out_ellipse.view(-1, 5), 0,
                                              pos_idcs)
        ellipse_targets_keep = torch.index_select(ellipse_targets.view(-1, 5),
                                                  0, pos_idcs)
        anchors_keep = torch.index_select(anchors.view(-1, 4), 0, pos_idcs)

        sigmas = (anchors_keep[:, 2] - anchors_keep[:, 0] + 1.0) / 2

        dx_o = out_ellipse_keep[:, 0]
        dy_o = out_ellipse_keep[:, 1]
        dl_o = out_ellipse_keep[:, 2]
        ds_o = out_ellipse_keep[:, 3]
        theta_o = torch.atan(out_ellipse_keep[:, 4])
        l_o = torch.exp(dl_o) * sigmas
        s_o = torch.exp(ds_o) * sigmas

        dx_t = ellipse_targets_keep[:, 0]
        dy_t = ellipse_targets_keep[:, 1]
        dl_t = ellipse_targets_keep[:, 2]
        ds_t = ellipse_targets_keep[:, 3]
        theta_t = torch.atan(ellipse_targets_keep[:, 4])

        dx = 2 * sigmas * (dx_o - dx_t)
        dy = 2 * sigmas * (dy_o - dy_t)
        dtheta = theta_o - theta_t

        trace = (torch.cos(dtheta) * torch.exp(dl_t - dl_o))**2 + \
                (torch.cos(dtheta) * torch.exp(ds_t - ds_o))**2 + \
                (torch.sin(dtheta) * torch.exp(dl_t - ds_o))**2 + \
                (torch.sin(dtheta) * torch.exp(ds_t - dl_o))**2

        dist = ((torch.cos(theta_o)*dx + torch.sin(theta_o)*dy) / l_o)**2 + \
               ((torch.cos(theta_o)*dy - torch.sin(theta_o)*dx) / s_o)**2

        determinant = 2 * (dl_o - dl_t) + 2 * (ds_o - ds_t)

        kld = (trace + dist + determinant - 2) / 2

        return kld.mean()
