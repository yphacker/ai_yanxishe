# coding=utf-8
# author=yphacker


import os
from transformers import BertTokenizer, AlbertTokenizer, XLMRobertaTokenizer, BartTokenizer

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
pretrain_model_path = os.path.join(data_path, "pretrain_model")
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")

model_path = os.path.join(data_path, "model")
for path in [model_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

pretrain_embedding = False
# pretrain_embedding = True


n_splits = 5
num_labels = 1
max_seq_len = 392

batch_size = 32
epochs_num = 2
train_print_step = 50
patience_epoch = 2

tokenizer_dict = {
    'bert': BertTokenizer,
    'bart': BartTokenizer,
}
