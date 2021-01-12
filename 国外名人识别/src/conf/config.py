# coding=utf-8
# author=yphacker

import os
import pandas as pd

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
model_path = os.path.join(data_path, "model")
submission_path = os.path.join(data_path, "submission")
for path in [model_path, submission_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

image_train_path = os.path.join(data_path, 'train')
image_test_path = os.path.join(data_path, 'test')
train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')


img_resize = 299
img_size = 256
n_splits = 5

batch_size = 32
epochs_num = 16
train_print_step = 100
patience_epoch = 5
adjust_lr_num = 0

train_df = pd.read_csv(train_path)
train_df = train_df[~train_df.filename.isin([83, 1212, 1340, 1679, 1734, 1985, 8930,
                                             9918, 10580, 10610, 11093])]
label_counts = train_df['label'].value_counts().rename('counts')
label_counts_df = label_counts.to_frame()
# label_list = label_counts_df[label_counts_df['counts'] >= 5].index.tolist()
label_list = label_counts_df.index.tolist()
# print(label_list)
# print(len(label_list))
label2id = {k: i for i, k in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}
num_classes = len(label_list)