# coding=utf-8
# author=yphacker

import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
model_path = os.path.join(data_path, "model")
model_submission_path = os.path.join(data_path, 'submission.csv')

origin_data_path = os.path.join(data_path, "origin_data")
origin_train_path = os.path.join(origin_data_path, 'sms_train.txt')
origin_test_path = os.path.join(origin_data_path, 'sms_test.txt')

process_data_path = os.path.join(data_path, "process_data")
train_path = os.path.join(process_data_path, 'train.csv')
test_path = os.path.join(process_data_path, 'test.csv')

train_check_path = os.path.join(data_path, 'train_check.txt')


max_seq_length = 75  # 序列长度
epochs_num = 8
batch_size = 32
print_per_batch = 10
improvement_step = print_per_batch * 10
num_labels = 2  # 类别数量
labels_dict = ['ham', 'spam']
