# coding=utf-8
# author=yphacker

import os
import cv2
import numpy as np
from mrcnn.utils import Dataset




class SafetyHelmetDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, img_path_list, box_list, mode='train'):
        self.mode = mode
        # define one class
        self.add_class("dataset", 1, "0")  # 0 - 'hat'
        self.add_class("dataset", 2, "1")  # 1 - 'person'
        for i in range(len(img_path_list)):
            img_path = img_path_list[i]
            img_id = int(img_path.split('/')[-1].split('.')[0])
            box = box_list[i]
            self.add_image("dataset", image_id=img_id, path=img_path, box=box)

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        boxes = info['box']
        boxes = boxes.split(' ')
        img_path = info['path']
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        # create one array for all masks, each on a different channel
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            box = box.split(',')
            row_s, row_e = int(box[1]), int(box[3])
            col_s, col_e = int(box[0]), int(box[2])
            masks[row_s:row_e, col_s:col_e, i] = 1
            label = box[4]
            class_ids.append(self.class_names.index(label))
        return masks, np.asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
