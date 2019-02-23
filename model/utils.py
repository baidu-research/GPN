import numpy as np


def iou(coord_a, coord_b):
    x1_a, y1_a, x2_a, y2_a = coord_a
    x1_b, y1_b, x2_b, y2_b = coord_b

    x_overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
    y_overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))

    intersection = x_overlap * y_overlap
    union = (x2_a - x1_a) * (y2_a - y1_a) + (x2_b - x1_b) * (y2_b - y1_b)\
        - intersection

    return intersection * 1.0 / union


def lr_schedule(lr, lr_factor, epoch_now, epoch_lr):
    """
    Learning rate schedule with respect to epoch
    lr: float, initial learning rate
    lr_factor: float, decreasing factor every epoch_lr
    epoch_now: int, the current epoch
    epoch_lr: int, decreasing every epoch_lr
    return: lr, float, scheduled learning rate.
    """
    return lr * np.power(lr_factor, epoch_now // epoch_lr)


def img_acc(out, coords, cfg, iou_thred=1e-5):
    batch_size, n_anchors, n_anchors, _, _ = out.shape
    n_scales = len(cfg['scales'])
    n_ratios = len(cfg['ratios'])
    out = out.reshape(batch_size, n_anchors, n_anchors, n_scales, n_ratios, 5)

    offset = (cfg['stride'] - 1.0) / 2
    x_anchors = np.arange(
        offset, offset + cfg['stride'] * (n_anchors - 1) + 1, cfg['stride']
    )
    y_anchors = np.array(x_anchors)

    idcs_max = np.argmax(out[..., 0].reshape(batch_size, -1), axis=1)
    idcs_x, idcs_y, idcs_s, idcs_r = np.unravel_index(
        idcs_max, (n_anchors, n_anchors, n_scales, n_ratios))

    acc = 0.0
    for i in range(batch_size):
        x_anchor = x_anchors[idcs_x[i]]
        y_anchor = y_anchors[idcs_y[i]]
        scale = cfg['scales'][idcs_s[i]]
        ratio = cfg['ratios'][idcs_r[i]]
        w_anchor = np.round(scale / np.sqrt(ratio))
        h_anchor = np.round(scale * np.sqrt(ratio))

        if cfg['use_regress']:
            out_i = out[i, idcs_x[i], idcs_y[i], idcs_s[i], idcs_r[i]]
            x_bbox = out_i[1] * w_anchor + x_anchor
            y_bbox = out_i[2] * h_anchor + y_anchor
            w_bbox = np.exp(out_i[3]) * w_anchor
            h_bbox = np.exp(out_i[4]) * h_anchor
        else:
            x_bbox = x_anchor
            y_bbox = y_anchor
            w_bbox = w_anchor
            h_bbox = h_anchor

        x1_bbox, y1_bbox = x_bbox - w_bbox / 2.0, y_bbox - h_bbox / 2.0
        x2_bbox, y2_bbox = x_bbox + w_bbox / 2.0, y_bbox + h_bbox / 2.0
        coord_bbox = (x1_bbox, y1_bbox, x2_bbox, y2_bbox)

        for coord in coords[i]:
            if tuple(coord) == (0, 0, 0, 0):
                break

            if iou(coord_bbox, coord) >= iou_thred:
                acc += 1.0
                break

    return acc / batch_size


def froc(froc_data, n_imgs, n_gt_boxes, iou_thred=0.5,
         fps_img=[0.5, 1, 2, 4, 8, 16]):
    M, N, _ = froc_data.shape
    scores = froc_data[:, :, 0].flatten()
    idcs_sorted = scores.argsort()[::-1]
    idcs_img, idcs_prop = np.unravel_index(idcs_sorted,
                                           dims=(M, N))

    tp = 0
    fp = 0
    tps = []
    fps = (np.sort(fps_img) * n_imgs).tolist()
    gt_boxes_hitted = set()

    # for each proposal sorted by their scores
    for i in range(len(idcs_sorted)):
        idx_img = idcs_img[i]
        idx_prop = idcs_prop[i]
        overlap = froc_data[idx_img, idx_prop, 1]
        gt_box_id = froc_data[idx_img, idx_prop, 2]

        # not hit
        if overlap < iou_thred:
            fp += 1

            if fp < fps[0]:
                continue

            tps.append(tp)
            fps.pop(0)

            if len(fps) == 0:
                break
        # hit
        else:
            # new hit
            if (idx_img, gt_box_id) not in gt_boxes_hitted:
                tp += 1
                gt_boxes_hitted.add((idx_img, gt_box_id))

    sens = np.array(tps) / n_gt_boxes
    FROC = sens.mean()

    return (FROC, sens)
