# coding=utf-8
# author=yphacker

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, transforms, mode='train'):
        self.transforms = transforms
        self.mode = mode
        self.imgs = [os.path.join(path, img) for img in os.listdir(path)]
        if self.mode == 'test':
            self.imgs = sorted(self.imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            self.imgs = np.random.permutation(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.mode == 'test':
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data.numpy().astype('float32'), label

    def __len__(self):
        return len(self.imgs)
