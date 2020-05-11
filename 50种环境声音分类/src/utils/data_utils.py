# coding=utf-8
# author=yphacker


import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from conf import config
from utils.data_augmentation import FixedRotation


class MyDataset(Dataset):
    def __init__(self, df, transform, mode='train'):
        self.df = df
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        if self.mode in ['train', 'val']:
            img = Image.open(self.df['filename'].iloc[index]).convert('RGB')
            img = self.transform(img)
            return img, torch.from_numpy(np.array(self.df['label'].iloc[index]))
        else:
            img = Image.open(self.df[index]).convert('RGB')
            img = self.transform(img)
            return img, torch.from_numpy(np.array(0))

    def __len__(self):
        return len(self.df)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomGrayscale(),
    # transforms.RandomRotation(90),
    FixedRotation([0, 90, 180, -90]),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize([config.img_size, config.img_size]),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize([config.img_size, config.img_size]),
    transforms.ToTensor(),
    normalize,
])
