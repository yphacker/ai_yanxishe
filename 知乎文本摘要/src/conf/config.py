# coding=utf-8
# author=yphacker

import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
pretrain_model_path = os.path.join(data_path, "pretrain_model")
train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')

model_path = os.path.join(data_path, "model")
submission_path = os.path.join(data_path, "submission")
for path in [model_path, submission_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

max_seq_len = 155
num_vocab = 41530

tokenizer = lambda x: x.split(' ')[:max_seq_len]
padding_idx = 0

batch_size = 32
epochs_num = 8

n_splits = 5
train_print_step = 500
patience_epoch = 1
