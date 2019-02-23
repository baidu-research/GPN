import os
from csv import reader

import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image

np.random.seed(0)


class DeepLesionDataset(Dataset):
    def __init__(self, cfg, mode):
        """
        mode =
        '1' : train;
        '2' : valid;
        '3' : test
        """
        self._cfg = dict(cfg)
        self._mode = mode
        self._preprocess()

    def _preprocess(self):
        anno_path = os.path.join(self._cfg['DATAPATH'], 'DL_info.csv')

        with open(anno_path) as f:
            lines = f.readlines()

        lines_iter = reader(lines)
        self._anno_header = next(lines_iter)
        self._anno_dict = {}
        self._img_names = []

        for line in lines_iter:
            mode = line[-1]
            noisy = line[10]

            if self._mode != mode:
                continue

            if noisy == '1':
                continue

            img_name = line[0]

            # one image may have multiple bounding boxes
            if img_name in self._anno_dict:
                self._anno_dict[img_name].append(line)
            else:
                self._anno_dict[img_name] = [line]
                self._img_names.append(img_name)

        self._n_imgs = len(self._img_names)

    def make_path(self, sub_path, slice_idx, delta):
        while not os.path.exists(
            os.path.join(self._cfg['DATAPATH'], 'Images_png', sub_path,
                         str(slice_idx + delta).zfill(3) + '.png')):
            delta -= np.sign(delta)
            if delta == 0:
                break

        return os.path.join(self._cfg['DATAPATH'], 'Images_png', sub_path,
                            str(slice_idx + delta).zfill(3) + '.png')

    def load_img(self, idx):
        img_name = self._img_names[idx]
        row = self._anno_dict[img_name][0]
        slice_intv = float(row[12].split(',')[2])
        patient_idx, study_idx, series_idx, slice_idx = img_name.split('_')
        sub_path = '_'.join([patient_idx, study_idx, series_idx])
        slice_idx = int(slice_idx.split('.')[0])
        img_cur = np.array(Image.open(
            self.make_path(sub_path, slice_idx, 0))).astype(np.int32)

        # find neighboring slices of img_cur
        rel_pos = float(self._cfg['SLICE_INTV']) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        rel_ceil = int(np.ceil(rel_pos))
        rel_floor = int(np.floor(rel_pos))
        rel_int = int(rel_pos)

        # required SLICE_INTV is a divisible to the actual slice_intv,
        # don't need interpolation
        if a == 0:
            img_prev = np.array(Image.open(
                self.make_path(sub_path, slice_idx, -rel_int))
            ).astype(np.int32)

            img_next = np.array(Image.open(
                self.make_path(sub_path, slice_idx, +rel_int))
            ).astype(np.int32)
        else:
            slice1 = np.array(Image.open(
                self.make_path(sub_path, slice_idx, -rel_ceil))
            ).astype(np.int32)
            slice2 = np.array(Image.open(
                self.make_path(sub_path, slice_idx, -rel_floor))
            ).astype(np.int32)
            img_prev = a * slice1 + b * slice2  # linear interpolation

            slice1 = np.array(Image.open(
                self.make_path(sub_path, slice_idx, +rel_ceil))
            ).astype(np.int32)
            slice2 = np.array(Image.open(
                self.make_path(sub_path, slice_idx, +rel_floor))
            ).astype(np.int32)
            img_next = a * slice1 + b * slice2  # linear interpolation

        img = np.stack([img_prev, img_cur, img_next], axis=2) - 32768

        return img

    def __len__(self):
        return self._n_imgs

    def __getitem__(self, idx):
        img_name = self._img_names[idx]
        img = self.load_img(idx)

        # bounding HU values to [0, 255]
        img = np.clip(
            img, self._cfg['HU_MIN'], self._cfg['HU_MAX']).astype(float)
        img = (img - self._cfg['HU_MIN']) / (
            self._cfg['HU_MAX'] - self._cfg['HU_MIN'])
        img = img * 255

        # initialize all the ground truth bounding boxes as
        # [-100, -100, -99, -99] so no anchors overlap
        gt_boxes = np.ones((self._cfg['MAX_NUM_GT_BOXES'], 4)) * -100.0
        gt_boxes[:, 2:4] += 1
        # initialize all the ground truth bounding ellipses as
        # [-100, -100, -99, -99, -100, -99, -99, -100] so no anchors overlap
        gt_ellipses = np.ones((self._cfg['MAX_NUM_GT_BOXES'], 8)) * -100.0
        gt_ellipses[:, 2:4] += 1
        gt_ellipses[:, 5:7] += 1
        spacing = 0
        num_gt_boxes = min(len(self._anno_dict[img_name]),
                           self._cfg['MAX_NUM_GT_BOXES'])
        # loading ground truth bounding boxes
        for i in range(num_gt_boxes):
            row = self._anno_dict[img_name][i]
            x1, y1, x2, y2 = map(float, row[6].split(','))
            gt_boxes[i] = np.array([x1, y1, x2, y2])
            x11, y11, x12, y12, x21, y21, x22, y22 = map(
                float, row[5].split(','))
            gt_ellipses[i] = np.array([x11, y11, x12, y12, x21, y21, x22, y22])
            # spacing is the same for different annotation of the same image
            spacing = float(row[12].split(',')[0])
        gt_boxes = np.array(gt_boxes)
        gt_boxes -= 1  # coordinates in info file start from 1
        gt_ellipses = np.array(gt_ellipses)
        gt_ellipses -= 1

        # resizing to cfg['MAX_SIZE'] based on spacing
        resize_factor = float(spacing) / self._cfg['NORM_SPACING']
        resized_shape = np.array(img.shape[:2]) * resize_factor
        if resized_shape[0] > self._cfg['MAX_SIZE']:
            resize_factor *= self._cfg['MAX_SIZE'] * 1.0 / resized_shape[0]
        if resize_factor != 1:
            img = cv2.resize(img, None, None, fx=resize_factor,
                             fy=resize_factor, interpolation=cv2.INTER_LINEAR)
            gt_boxes *= resize_factor
            gt_ellipses *= resize_factor

        # subtract mean
        img -= self._cfg['PIXEL_MEANS']

        # pad 0 to cfg['MAX_SIZE']
        resized_shape = img.shape
        img_max = np.zeros((self._cfg['MAX_SIZE'], self._cfg['MAX_SIZE'], 3))
        img_max[:resized_shape[0], :resized_shape[1], :] = img
        img = img_max.transpose((2, 0, 1)).astype(np.float32)
        gt_boxes = gt_boxes.astype(np.float32)
        gt_ellipses = gt_ellipses.astype(np.float32)

        return (img, gt_boxes, gt_ellipses)
