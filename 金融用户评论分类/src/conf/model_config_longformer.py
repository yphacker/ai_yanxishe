# coding=utf-8
# author=yphacker

import os
from conf import config
from transformers import RobertaTokenizer

# pretrain_model_name = 'longformer-base-4096'
pretrain_model_name = 'longformer-large-40966'
pretrain_model_path = os.path.join(config.pretrain_model_path, pretrain_model_name)

pretrain_model_name = 'roberta-base'
tokenizer_path = os.path.join(config.pretrain_model_path, pretrain_model_name)
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

learning_rate = 2e-5
adjust_lr_num = 0
