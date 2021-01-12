# coding=utf-8
# author=yphacker

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from conf import config
import pandas as pd

train_df = pd.read_csv('../data/train.csv')
train_df['filename'] = train_df['filename'].apply(lambda x: '../data/train/{0}.jpg'.format(x))
for i in range(train_df.shape[0]):
    try:
        img = Image.open(train_df['filename'].iloc[i]).convert('RGB')
    except:
        print(i)


