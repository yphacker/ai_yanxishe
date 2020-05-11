# coding=utf-8
# author=yphacker


import os
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

pretrain_embedding = False
# pretrain_embedding = True
# embed_dim = 300
max_seq_len = 222

# tokenizer = lambda x: x.split(' ')[:max_seq_len]
# padding_idx = 0


num_labels = 3
n_splits = 5

batch_size = 32
epochs_num = 8
train_print_step = 10
patience_epoch = 3

label2id = {
    'AGAINST': 0,
    'FAVOR': 1,
    'NONE': 2
}
id2label = {v: k for k, v in label2id.items()}

tokenizer_dict = {
    'bert': BertTokenizer,
}
