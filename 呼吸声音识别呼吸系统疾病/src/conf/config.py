# coding=utf-8
# author=yphacker

import os
import sys

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
model_path = os.path.join(data_path, 'model')

train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')

MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')

num_labels = 8
batch_size = 64
epochs_num = 8

train_print_step = 10
patience_epoch = 5
