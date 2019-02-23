import sys
import os
import argparse
import logging
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import model.utils as utils  # noqa
import model.functional as functional  # noqa
from data.deep_lesion_dataset import DeepLesionDataset  # noqa
from model.gpn import GPN  # noqa
from model.bbox_transform import bbox_overlaps  # noqa
from model.ellipse_transform import ellipse_overlaps  # noqa


parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=1, type=int, help='Number of'
                    ' workers for each data loader, default 1')
parser.add_argument('--iou_thred', default=0.5, type=float, help='IoU'
                    ' threshold, default 0.5')
parser.add_argument('--fps_img', default='0.5,1,2,4,8,16', type=str,
                    help='False positives per image, default 0.5,1,2,4,8,16')


def run(args):
    with open(os.path.join(args.save_path, 'cfg.json')) as f:
        cfg = json.load(f)

    model = GPN(cfg).cuda().eval()
    ckpt_path = os.path.join(args.save_path, 'best.ckpt')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])

    dataloader = DataLoader(DeepLesionDataset(cfg, '3'),
                            batch_size=cfg['TEST.IMS_PER_BATCH'],
                            num_workers=args.num_workers,
                            drop_last=False)

    steps = len(dataloader)
    dataiter = iter(dataloader)

    time_now = time.time()
    loss_sum = 0
    loss_cls_sum = 0
    loss_ellipse_sum = 0
    n_proposals_sum = 0
    acc_pos_sum = 0
    acc_neg_sum = 0
    angle_err_sum = 0
    n_imgs = 0
    n_gt_boxes = 0
    froc_data_sum = np.zeros((steps, dataloader.batch_size,
                              cfg['TEST.RPN_POST_NMS_TOP_N'], 3))
    im_info = (cfg['MAX_SIZE'], cfg['MAX_SIZE'])
    for step in range(steps):
        img, gt_boxes, gt_ellipses = next(dataiter)
        gt_boxes = gt_boxes.cuda(async=True)
        gt_ellipses = gt_ellipses.cuda(async=True)
        labels, bbox_targets, ellipse_targets = model.ellipse_target(
            gt_boxes, gt_ellipses)

        img = Variable(img.cuda(async=True))
        labels = Variable(labels.cuda(async=True))
        ellipse_targets = Variable(ellipse_targets.cuda(async=True))

        out_cls, out_ellipse = model(img)
        loss_cls = model.loss_cls(out_cls, labels)
        loss_ellipse = model.loss_ellipse(out_ellipse, labels, ellipse_targets)
        loss = loss_cls + loss_ellipse
        acc_pos, acc_neg = functional.acc(out_cls, labels)
        n_proposals = functional.n_proposals(out_cls)
        angle_err = functional.angle_err(out_ellipse, labels, ellipse_targets)

        froc_data_batch = np.zeros((dataloader.batch_size,
                                    cfg['TEST.RPN_POST_NMS_TOP_N'], 3))
        for i in range(out_cls.size(0)):
            # final proposals and scores after nms for each image
            boxes, ellipses, scores = model.ellipse_proposal(out_cls[i],
                                                             out_ellipse[i])
            # keep non-padded gt_boxes/gt_ellipses
            # keep = gt_boxes[i].gt(0).sum(dim=1).nonzero().view(-1)
            # overlaps = bbox_overlaps(boxes, gt_boxes[i][keep])
            keep = gt_ellipses[i].gt(0).sum(dim=1).nonzero().view(-1)
            overlaps = ellipse_overlaps(ellipses, gt_ellipses[i][keep],
                                        im_info)
            overlaps_max, idcs_max = overlaps.max(dim=1)
            n_ = scores.size(0)
            n_imgs += 1
            n_gt_boxes += keep.size(0)

            froc_data_batch[i, :n_, 0] = scores.cpu().numpy()
            froc_data_batch[i, :n_, 1] = overlaps_max.cpu().numpy()
            froc_data_batch[i, :n_, 2] = idcs_max.cpu().numpy()

        loss_sum += loss.data[0]
        loss_cls_sum += loss_cls.data[0]
        loss_ellipse_sum += loss_ellipse.data[0]
        n_proposals_sum += n_proposals.data[0]
        acc_pos_sum += acc_pos.data[0]
        acc_neg_sum += acc_neg.data[0]
        angle_err_sum += angle_err.data[0]
        froc_data_sum[step] = froc_data_batch

        if step % cfg['log_every'] == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg['log_every']
            loss_cls_sum /= cfg['log_every']
            loss_ellipse_sum /= cfg['log_every']
            n_proposals_sum = int(n_proposals_sum / cfg['log_every'])
            acc_pos_sum /= cfg['log_every']
            acc_neg_sum /= cfg['log_every']
            angle_err_sum /= cfg['log_every']

            logging.info(
                '{}, Test, Step : {}, Total Loss : {:.4f}, Cls Loss : {:.4f}, '
                'Ellipse Loss : {:.4f}, Pos Acc : {:.3f}, Neg Acc : {:.3f}, '
                'Angle Err : {:.3f}, #Props/Img : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"), step, loss_sum,
                        loss_cls_sum, loss_ellipse_sum, acc_pos_sum,
                        acc_neg_sum, angle_err_sum, n_proposals_sum,
                        time_spent))

            loss_sum = 0
            loss_cls_sum = 0
            loss_ellipse_sum = 0
            n_proposals_sum = 0
            acc_pos_sum = 0.0
            acc_neg_sum = 0.0
            angle_err_sum = 0

    fps_img = list(map(float, args.fps_img.split(',')))
    FROC, sens = utils.froc(
        froc_data_sum.reshape((-1, cfg['TEST.RPN_POST_NMS_TOP_N'], 3)),
        n_imgs, n_gt_boxes, iou_thred=args.iou_thred, fps_img=fps_img)
    sens_str = '\t'.join(list(map(lambda x: '{:.3f}'.format(x), sens)))
    fps_img_str = '\t'.join(list(map(lambda x: '{:.2f}'.format(x), fps_img)))

    print('*'*10 + 'False/Image' + '*'*10)
    print(fps_img_str)
    print('*'*10 + 'Sensitivity' + '*'*10)
    print(sens_str)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
