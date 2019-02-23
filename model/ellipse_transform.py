import torch

import numpy as np


def ellipse2box(ellipses, pad):
    """
    ellipses: (N, 8) ndarray of float
    pad: int, padded margin with respect to the corner of ellipse
    """
    xs = torch.stack(
        (ellipses[:, 0], ellipses[:, 2], ellipses[:, 4], ellipses[:, 6]), 1)
    ys = torch.stack(
        (ellipses[:, 1], ellipses[:, 3], ellipses[:, 5], ellipses[:, 7]), 1)
    x_min, _ = torch.min(xs, dim=1)
    x_max, _ = torch.max(xs, dim=1)
    y_min, _ = torch.min(ys, dim=1)
    y_max, _ = torch.max(ys, dim=1)

    boxes = torch.Tensor(ellipses.size(0), 4).fill_(0).type_as(ellipses)

    # x1
    boxes[:, 0] = x_min - pad
    # y1
    boxes[:, 1] = y_min - pad
    # x2
    boxes[:, 2] = x_max + pad
    # y2
    boxes[:, 3] = y_max + pad

    return boxes


def ellipse_mask(ellipses, im_info):
    """
    ellipses: (N, 8) ndarray of float, ellipses
    im_info: 2d int tuple, (width, height)
    mask: (width, height, N) ndarray of binary mask representing if pixels
        are within the corresponding ellipse.
    """
    N = ellipses.size(0)
    width, height = im_info

    longs_x = (torch.abs(ellipses[:, 2] - ellipses[:, 0]) + 1.0) / 2.0
    longs_y = (torch.abs(ellipses[:, 3] - ellipses[:, 1]) + 1.0) / 2.0
    longs = torch.sqrt(longs_x**2 + longs_y**2)
    shorts_x = (torch.abs(ellipses[:, 6] - ellipses[:, 4]) + 1.0) / 2.0
    shorts_y = (torch.abs(ellipses[:, 7] - ellipses[:, 5]) + 1.0) / 2.0
    shorts = torch.sqrt(shorts_x**2 + shorts_y**2)
    ctr_x = torch.min(ellipses[:, 0], ellipses[:, 2]) + (longs_x - 0.5)
    ctr_y = torch.min(ellipses[:, 1], ellipses[:, 3]) + (longs_y - 0.5)
    # tan(theta), theta is the angle between x > 0 axis and long > 0 axis.
    sign = torch.sign(
        (ellipses[:, 2] - ellipses[:, 0]) * (ellipses[:, 3] - ellipses[:, 1]))
    tan = sign * longs_y / longs_x
    theta = torch.atan(tan)

    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)
    x = torch.from_numpy(x.ravel()).type_as(ellipses)
    y = torch.from_numpy(y.ravel()).type_as(ellipses)

    x, y = x.view(-1, 1), y.view(-1, 1)

    ctr_x, ctr_y = ctr_x.view(1, -1), ctr_y.view(1, -1)
    longs, shorts = longs.view(1, -1), shorts.view(1, -1)
    theta = theta.view(1, -1)

    dx = x - ctr_x
    dy = y - ctr_y

    dist = ((torch.cos(theta)*dx + torch.sin(theta)*dy) / longs)**2 + \
           ((torch.cos(theta)*dy - torch.sin(theta)*dx) / shorts)**2

    mask = dist.le(1.0).view(width, height, N)

    return mask


def ellipse_overlaps(qr_ellipses, gt_ellipses, im_info):
    """
    qr_ellipses: (N, 8) ndarray of float, query ellipses
    gt_ellipses: (K, 8) ndarray of float, ground truth ellipses
    im_info: 2d int tuple, (width, height)
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = qr_ellipses.size(0)
    K = gt_ellipses.size(0)
    width, height = im_info

    qr_mask = ellipse_mask(qr_ellipses, im_info)
    gt_mask = ellipse_mask(gt_ellipses, im_info)

    qr_mask = qr_mask.view(-1, N).transpose(1, 0).float()
    gt_mask = gt_mask.view(-1, K).float()

    I = torch.matmul(qr_mask, gt_mask)
    U = width * height - torch.matmul(1 - qr_mask, 1 - gt_mask)
    overlaps = I / U

    return overlaps


def ellipse_transform(ex_rois, gt_rois):
    """
    ex_rois: (N, 4) ndarray of float, anchors
    gt_rois: (N, 8) ndarray of float, ellipses
    """
    # assuming anchors have equal width and height, which is defined as sigma*2
    ex_sigmas = (ex_rois[:, 2] - ex_rois[:, 0] + 1.0) / 2
    ex_ctr_x = ex_rois[:, 0] + (ex_sigmas - 0.5)
    ex_ctr_y = ex_rois[:, 1] + (ex_sigmas - 0.5)

    gt_longs_x = (torch.abs(gt_rois[:, 2] - gt_rois[:, 0]) + 1.0) / 2.0
    gt_longs_y = (torch.abs(gt_rois[:, 3] - gt_rois[:, 1]) + 1.0) / 2.0
    gt_longs = torch.sqrt(gt_longs_x**2 + gt_longs_y**2)
    gt_shorts_x = (torch.abs(gt_rois[:, 6] - gt_rois[:, 4]) + 1.0) / 2.0
    gt_shorts_y = (torch.abs(gt_rois[:, 7] - gt_rois[:, 5]) + 1.0) / 2.0
    gt_shorts = torch.sqrt(gt_shorts_x**2 + gt_shorts_y**2)
    gt_ctr_x = torch.min(gt_rois[:, 0], gt_rois[:, 2]) + (gt_longs_x - 0.5)
    gt_ctr_y = torch.min(gt_rois[:, 1], gt_rois[:, 3]) + (gt_longs_y - 0.5)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_sigmas * 2)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_sigmas * 2)
    targets_dl = torch.log(gt_longs / ex_sigmas)
    targets_ds = torch.log(gt_shorts / ex_sigmas)
    # tan(theta), theta is the angle between x > 0 axis and long > 0 axis.
    targets_sign = torch.sign(
        (gt_rois[:, 2] - gt_rois[:, 0]) * (gt_rois[:, 3] - gt_rois[:, 1]))
    targets_tan = targets_sign * gt_longs_y / gt_longs_x

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dl, targets_ds, targets_tan), 1)

    return targets


def ellipse_transform_inv(ellipses, deltas):
    sigmas = (ellipses[:, 2] - ellipses[:, 0] + 1.0) / 2
    ctr_x = ellipses[:, 0] + (sigmas - 0.5)
    ctr_y = ellipses[:, 1] + (sigmas - 0.5)

    dx = deltas[:, 0::5]
    dy = deltas[:, 1::5]
    dl = deltas[:, 2::5]
    ds = deltas[:, 3::5]
    tan = deltas[:, 4::5]
    theta = torch.atan(tan)
    sign = torch.sign(theta)

    pred_ctr_x = dx * sigmas.unsqueeze(1) * 2 + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * sigmas.unsqueeze(1) * 2 + ctr_y.unsqueeze(1)

    pred_l = torch.exp(dl) * sigmas.unsqueeze(1)
    pred_s = torch.exp(ds) * sigmas.unsqueeze(1)
    pred_l_x = torch.abs(pred_l * torch.cos(theta))
    pred_l_y = torch.abs(pred_l * torch.sin(theta))
    pred_s_x = torch.abs(pred_s * torch.sin(theta))
    pred_s_y = torch.abs(pred_s * torch.cos(theta))

    pred_elipses = torch.Tensor(deltas.size(0), 8).fill_(0).type_as(deltas)

    # x11
    pred_elipses[:, 0::8] = pred_ctr_x - (pred_l_x - 0.5)
    # y11
    pred_elipses[:, 1::8] = pred_ctr_y - sign * (pred_l_y - 0.5)
    # x12
    pred_elipses[:, 2::8] = pred_ctr_x + (pred_l_x - 0.5)
    # y12
    pred_elipses[:, 3::8] = pred_ctr_y + sign * (pred_l_y - 0.5)
    # x21
    pred_elipses[:, 4::8] = pred_ctr_x - (pred_s_x - 0.5)
    # y21
    pred_elipses[:, 5::8] = pred_ctr_y + sign * (pred_s_y - 0.5)
    # x22
    pred_elipses[:, 6::8] = pred_ctr_x + (pred_s_x - 0.5)
    # y22
    pred_elipses[:, 7::8] = pred_ctr_y - sign * (pred_s_y - 0.5)

    return pred_elipses
