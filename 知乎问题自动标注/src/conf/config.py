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

# max_seq_len = 140
max_seq_len = 256

topic2id_df = pd.read_csv('{}/{}'.format(data_path, 'topic2id.csv'))
num_labels = topic2id_df.shape[0]
n_splits = 5

batch_size = 32
epochs_num = 8
train_print_step = 100
patience_epoch = 2

tokenizer_dict = {
    'bert': BertTokenizer,
}
