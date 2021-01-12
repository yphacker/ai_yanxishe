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
train_df['label'] = train_df['label'].apply(lambda x: config.label2id[x] if x in config.label2id else 0)
for i in range(train_df.shape[0]):
    try:
        img = Image.open(train_df['filename'].iloc[i]).convert('RGB')
    except:
        print(i)

# train_data = pd.DataFrame(columns=train_df.columns)
# val_data = pd.DataFrame(columns=train_df.columns)
# random_state = 0
# for i, label in enumerate(config.label_list):
#     tmp = train_df[train_df['label'] == label]
#     num = 2 if tmp.shape[0] >= 2 else 1
#     train_data = train_data.append(tmp.sample(n=num, random_state=random_state), ignore_index=True)
#     random_state += 1
#     val_data = val_data.append(tmp.sample(n=num, random_state=random_state), ignore_index=True)
#     random_state += 1
