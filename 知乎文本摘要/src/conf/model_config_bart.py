# coding=utf-8
# author=yphacker

import os
from conf import config

pretrain_model_name = 'bart-large-cnn'
pretrain_model_path = os.path.join(config.pretrain_model_path, pretrain_model_name)

learning_rate = 1e-5
adjust_lr_num = 1
