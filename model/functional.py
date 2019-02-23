import torch
import torch.nn.functional as F
import numpy as np


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.7):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        scores: (N) FloatTensor
        boxes: (N, 4) FloatTensor
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
    Return:
        The indices of the kept boxes with respect to N.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(dim=0, descending=True)  # sort in ascending order
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[0]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[1:]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]

    keep = keep[:count]

    return keep


def n_proposals(out_cls):
    vals, idcs = out_cls.view(-1, 2).max(1)
    n_proposals = idcs.eq(1).type(torch.cuda.FloatTensor).sum() / len(out_cls)

    return n_proposals


def acc(out_cls, labels):
    pos_idcs = labels.view(-1).eq(1).nonzero().view(-1)
    out_cls_pos = torch.index_select(out_cls.view(-1, 2), 0, pos_idcs)
    prob_pos = F.softmax(out_cls_pos, dim=1)[:, 1]
    acc_pos = prob_pos.ge(0.5).type(
        torch.cuda.FloatTensor).sum() / len(prob_pos)

    neg_idcs = labels.view(-1).eq(0).nonzero().view(-1)
    out_cls_neg = torch.index_select(out_cls.view(-1, 2), 0, neg_idcs)
    prob_neg = F.softmax(out_cls_neg, dim=1)[:, 0]
    acc_neg = prob_neg.ge(0.5).type(
        torch.cuda.FloatTensor).sum() / len(prob_neg)

    return (acc_pos, acc_neg)


def angle_err(out_ellipse, labels, ellipse_targets):
    pos_idcs = labels.view(-1).eq(1).nonzero().view(-1)
    out_ellipse_keep = torch.index_select(out_ellipse.view(-1, 5), 0, pos_idcs)
    ellipse_targets_keep = torch.index_select(ellipse_targets.view(-1, 5), 0,
                                              pos_idcs)

    out_tan = out_ellipse_keep[:, 4]
    out_angle = torch.atan(out_tan) * 180 / np.pi
    targets_tan = ellipse_targets_keep[:, 4]
    targets_angle = torch.atan(targets_tan) * 180 / np.pi

    err = torch.abs(out_angle - targets_angle).sum() / len(out_angle)

    return err
