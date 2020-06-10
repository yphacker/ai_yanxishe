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

# max_seq_len = 256
max_seq_len = 49

n_splits = 5

batch_size = 32
epochs_num = 8
train_print_step = 100
patience_epoch = 2

label2id_list = ['OTHERS', 'music.play', 'navigation.navigation', 'phone_call.make_a_phone_call',
                 'navigation.cancel_navigation', 'music.pause', 'navigation.open', 'music.next',
                 'navigation.start_navigation', 'navigation.start_navigation', 'phone_call.cancel', 'music.prev']
label2id = {v: k for k, v in enumerate(label2id_list)}
id2label = {k: v for k, v in enumerate(label2id_list)}

num_labels = len(label2id_list)

tokenizer_dict = {
    'bert': BertTokenizer,
}
