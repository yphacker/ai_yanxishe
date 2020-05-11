# coding=utf-8
# author=yphacker

import os
from conf import config

bert_data_path = os.path.join(config.data_path, 'uncased_L-12_H-768_A-12')
bert_config_path = os.path.join(bert_data_path, 'bert_config.json')
bert_checkpoint_path = os.path.join(bert_data_path, 'bert_model.ckpt')
bert_vocab_path = os.path.join(bert_data_path, 'vocab.txt')


learning_rate = 1e-5
grad_clip = 5.0
