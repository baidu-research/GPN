import sys
import os
import argparse
import logging
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import model.utils as utils  # noqa
import model.functional as functional  # noqa
from data.deep_lesion_dataset import DeepLesionDataset  # noqa
from model.gpn import GPN  # noqa
from model.bbox_transform import bbox_overlaps  # noqa
from model.ellipse_transform import ellipse_overlaps  # noqa


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved models')
parser.add_argument('--num_workers', default=1, type=int, help='''Number of
                    workers for each data loader''')
parser.add_argument('--resume', default=0, type=int, help='''If resume from
                    previous run''')


def train_epoch(summary, summary_writer, cfg, model, optimizer, dataloader):
    model.train()

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
    froc_idx = 0
    froc_data_sum = np.zeros((cfg['log_every'] // cfg['TRAIN.FROC_EVERY'],
                              dataloader.batch_size,
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
        # number of raw proposals with prob > 0.5
        n_proposals = functional.n_proposals(out_cls)
        angle_err = functional.angle_err(out_ellipse, labels, ellipse_targets)

        loss_sum += loss.data[0]
        loss_cls_sum += loss_cls.data[0]
        loss_ellipse_sum += loss_ellipse.data[0]
        n_proposals_sum += n_proposals.data[0]
        acc_pos_sum += acc_pos.data[0]
        acc_neg_sum += acc_neg.data[0]
        angle_err_sum += angle_err.data[0]

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), cfg['grad_norm'])
        optimizer.step()

        summary['step'] += 1

        if summary['step'] % cfg['TRAIN.FROC_EVERY'] == 0:
            froc_data_batch = np.zeros((dataloader.batch_size,
                                        cfg['TEST.RPN_POST_NMS_TOP_N'], 3))
            for i in range(out_cls.size(0)):
                # final proposals and scores after nms for each image
                boxes, ellipses, scores = model.ellipse_proposal(
                    out_cls[i], out_ellipse[i])
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

            froc_data_sum[froc_idx] = froc_data_batch
            froc_idx += 1

        if summary['step'] % cfg['log_every'] == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg['log_every']
            loss_cls_sum /= cfg['log_every']
            loss_ellipse_sum /= cfg['log_every']
            n_proposals_sum = int(n_proposals_sum / cfg['log_every'])
            acc_pos_sum /= cfg['log_every']
            acc_neg_sum /= cfg['log_every']
            angle_err_sum /= cfg['log_every']
            FROC, sens = utils.froc(
                froc_data_sum.reshape((-1, cfg['TEST.RPN_POST_NMS_TOP_N'], 3)),
                n_imgs, n_gt_boxes, iou_thred=cfg['TEST.FROC_OVERLAP'])
            sens_str = ' '.join(list(map(lambda x: '{:.3f}'.format(x), sens)))

            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Total Loss : {:.4f}, '
                'Cls Loss : {:.4f}, Ellipse Loss : {:.4f}, Pos Acc : {:.3f}, '
                'Neg Acc : {:.3f}, Angle Err : {:.3f}, FROC : {:.3f}, '
                'Sens : {}, #Props/Img : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], loss_sum,
                        loss_cls_sum, loss_ellipse_sum, acc_pos_sum,
                        acc_neg_sum, angle_err_sum, FROC, sens_str,
                        n_proposals_sum, time_spent))

            summary_writer.add_scalar(
                'train/total_loss', loss_sum, summary['step'])
            summary_writer.add_scalar(
                'train/loss_cls', loss_cls_sum, summary['step'])
            summary_writer.add_scalar(
                'train/loss_ellipse', loss_ellipse_sum, summary['step'])
            summary_writer.add_scalar(
                'train/n_proposals', n_proposals_sum, summary['step'])
            summary_writer.add_scalar(
                'train/acc_pos', acc_pos_sum, summary['step'])
            summary_writer.add_scalar(
                'train/acc_neg', acc_neg_sum, summary['step'])
            summary_writer.add_scalar(
                'train/angle_err', angle_err_sum, summary['step'])
            summary_writer.add_scalar(
                'train/FROC', FROC, summary['step'])

            loss_sum = 0
            loss_cls_sum = 0
            loss_ellipse_sum = 0
            n_proposals_sum = 0
            acc_pos_sum = 0.0
            acc_neg_sum = 0.0
            angle_err_sum = 0
            n_imgs = 0
            n_gt_boxes = 0
            froc_idx = 0
            froc_data_sum = np.zeros((
                cfg['log_every'] // cfg['TRAIN.FROC_EVERY'],
                dataloader.batch_size, cfg['TEST.RPN_POST_NMS_TOP_N'], 3))

    summary['epoch'] += 1

    return summary


def valid_epoch(summary, cfg, model, dataloader):
    model.eval()

    steps = len(dataloader)
    dataiter = iter(dataloader)

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

    FROC, sens = utils.froc(
        froc_data_sum.reshape((-1, cfg['TEST.RPN_POST_NMS_TOP_N'], 3)),
        n_imgs, n_gt_boxes, iou_thred=cfg['TEST.FROC_OVERLAP'])
    sens_str = ' '.join(list(map(lambda x: '{:.3f}'.format(x), sens)))

    summary['loss'] = loss_sum / steps
    summary['loss_cls'] = loss_cls_sum / steps
    summary['loss_ellipse'] = loss_ellipse_sum / steps
    summary['n_proposals'] = int(n_proposals_sum / steps)
    summary['acc_pos'] = acc_pos_sum / steps
    summary['acc_neg'] = acc_neg_sum / steps
    summary['angle_err'] = angle_err_sum / steps
    summary['FROC'] = FROC
    summary['sens_str'] = sens_str

    return summary


def run(args):
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    model = GPN(cfg).cuda()
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                    weight_decay=cfg['weight_decay'])

    dataloader_train = DataLoader(DeepLesionDataset(cfg, '1'),
                                  batch_size=cfg['TRAIN.IMS_PER_BATCH'],
                                  num_workers=args.num_workers,
                                  drop_last=False,
                                  shuffle=True)
    dataloader_valid = DataLoader(DeepLesionDataset(cfg, '2'),
                                  batch_size=cfg['TEST.IMS_PER_BATCH'],
                                  num_workers=args.num_workers,
                                  drop_last=False)

    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'loss_cls': float('inf'),
                     'loss_ellipse': float('inf')}
    summary_writer = SummaryWriter(args.save_path)
    FROC_valid_best = 0
    epoch_start = 0

    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'train.ckpt')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        FROC_valid_best = ckpt['FROC_valid_best']
        epoch_start = ckpt['epoch']

    for epoch in range(epoch_start, cfg['epoch']):
        lr = utils.lr_schedule(cfg['lr'], cfg['lr_factor'],
                               summary_train['epoch'], cfg['lr_epoch'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        summary_train = train_epoch(summary_train, summary_writer, cfg, model,
                                    optimizer, dataloader_train)

        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, cfg, model,
                                    dataloader_valid)
        time_spent = time.time() - time_now

        logging.info(
            '{}, Valid, Epoch : {}, Step : {}, Total Loss : {:.4f}, '
            'Cls Loss : {:.4f}, Ellipse Loss : {:.4f}, Pos Acc : {:.3f}, '
            'Neg Acc : {:.3f}, Angle Err : {:.3f}, FROC : {:.3f}, Sens : {}, '
            '#Props/Img : {}, Run Time : {:.2f} sec'
            .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['epoch'], summary_train['step'],
                    summary_valid['loss'], summary_valid['loss_cls'],
                    summary_valid['loss_ellipse'], summary_valid['acc_pos'],
                    summary_valid['acc_neg'], summary_valid['angle_err'],
                    summary_valid['FROC'], summary_valid['sens_str'],
                    summary_valid['n_proposals'], time_spent))

        summary_writer.add_scalar('valid/total_loss',
                                  summary_valid['loss'],
                                  summary_train['step'])
        summary_writer.add_scalar('valid/loss_cls',
                                  summary_valid['loss_cls'],
                                  summary_train['step'])
        summary_writer.add_scalar('valid/loss_ellipse',
                                  summary_valid['loss_ellipse'],
                                  summary_train['step'])
        summary_writer.add_scalar('valid/n_proposals',
                                  summary_valid['n_proposals'],
                                  summary_train['step'])
        summary_writer.add_scalar('valid/acc_pos',
                                  summary_valid['acc_pos'],
                                  summary_train['step'])
        summary_writer.add_scalar('valid/acc_neg',
                                  summary_valid['acc_neg'],
                                  summary_train['step'])
        summary_writer.add_scalar('valid/angle_err',
                                  summary_valid['angle_err'],
                                  summary_train['step'])
        summary_writer.add_scalar('valid/FROC',
                                  summary_valid['FROC'],
                                  summary_train['step'])

        if summary_valid['FROC'] > FROC_valid_best:
            FROC_valid_best = summary_valid['FROC']

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'FROC_valid_best': FROC_valid_best,
                        'state_dict': model.state_dict()},
                       os.path.join(args.save_path, 'best.ckpt'))

            logging.info(
                '{}, Best, Epoch : {}, Step : {}, Total Loss : {:.4f}, '
                'Cls Loss : {:.4f}, Ellipse Loss : {:.4f}, Pos Acc : {:.3f}, '
                'Neg Acc : {:.3f}, Angle Err : {:.3f}, FROC : {:.3f}, '
                'Sens : {}, #Props/Img : {}'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary_train['epoch'], summary_train['step'],
                        summary_valid['loss'], summary_valid['loss_cls'],
                        summary_valid['loss_ellipse'],
                        summary_valid['acc_pos'], summary_valid['acc_neg'],
                        summary_valid['angle_err'], summary_valid['FROC'],
                        summary_valid['sens_str'],
                        summary_valid['n_proposals']
                        ))

        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'FROC_valid_best': FROC_valid_best,
                    'state_dict': model.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))

    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
