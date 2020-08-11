# coding=utf-8
# author=yphacker


import os
import pandas as pd
from transformers import BertTokenizer

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
word_embedding_path = os.path.join(data_path, "glove.840B.300d.txt")
pretrain_model_path = os.path.join(data_path, "pretrain_model")
pretrain_embedding_path = os.path.join(data_path, "pretrain_embedding.npz")
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")
model_path = os.path.join(data_path, "model")
submission_path = os.path.join(data_path, "submission")
for path in [model_path, submission_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

max_seq_len = 100
n_splits = 5

batch_size = 32
epochs_num = 8
train_print_step = 50
patience_epoch = 2

tokenizer_dict = {
    'bert': BertTokenizer,
}

label2id = {'Other': 0, 'Entity-Destination(e1,e2)': 1, 'Cause-Effect(e2,e1)': 2, 'Member-Collection(e2,e1)': 3,
            'Entity-Origin(e1,e2)': 4, 'Message-Topic(e1,e2)': 5, 'Component-Whole(e2,e1)': 6,
            'Component-Whole(e1,e2)': 7, 'Instrument-Agency(e2,e1)': 8, 'Content-Container(e1,e2)': 9,
            'Product-Producer(e2,e1)': 10, 'Cause-Effect(e1,e2)': 11, 'Product-Producer(e1,e2)': 12,
            'Content-Container(e2,e1)': 13, 'Message-Topic(e2,e1)': 14, 'Entity-Origin(e2,e1)': 15,
            'Instrument-Agency(e1,e2)': 16, 'Member-Collection(e1,e2)': 17}
id2label = {v: k for k, v in label2id.items()}

num_labels = len(label2id)
