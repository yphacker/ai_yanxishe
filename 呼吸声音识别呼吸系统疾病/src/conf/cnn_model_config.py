# coding=utf-8
# author=yphacker

import os
from conf import config

save_path = os.path.join(config.model_path, 'cnn')
if not os.path.isdir(save_path):
    os.makedirs(save_path)
model_save_path = os.path.join(save_path, 'best')
model_submission_path = os.path.join(config.data_path, 'cnn_submission.csv')