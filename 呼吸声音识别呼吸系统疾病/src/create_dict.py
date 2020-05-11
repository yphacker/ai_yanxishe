# -*- coding: utf-8 -*
import json
import os
import pandas
from conf import config

special_chars = ['_PAD_', '_EOS_', '_SOS_', '_UNK_']  # '_START_'
_PAD_ = 0
_EOS_ = 1
_UNK_ = 2
_SOS_ = 3


# _START_ = 3
def create(filein, LABEL_PATH):
    print('save to', LABEL_PATH)
    word_dict = dict()
    for i, word in enumerate(special_chars):
        word_dict[word] = i
    label_dict = dict()
    f = pandas.read_csv(filein, usecols=['label'])
    labels = f.values[:, 0]
    labels = labels.astype('str')
    for label in labels:
        if label not in label_dict:
            label_dict[label] = len(label_dict)
    with open(os.path.join(LABEL_PATH), 'w', encoding='utf-8') as fout:
        json.dump(label_dict, fout)
    print('build dict done.')


def load_dict():
    char_dict_re = dict()
    dict_path = os.path.join(config.data_path, 'word.dict')
    with open(dict_path, encoding='utf-8') as fin:
        char_dict = json.load(fin)
    for k, v in char_dict.items():
        char_dict_re[v] = k
    return char_dict, char_dict_re


def load_label_dict():
    char_dict_re = dict()
    char_dict = {"Healthy": 0, "COPD": 1, "LRTI": 2, "URTI": 3, "Bronchiectasis": 4,
                 "Pneumonia": 5, "Bronchiolitis": 6, "Asthma": 7}
    for k, v in char_dict.items():
        char_dict_re[v] = k
    return char_dict, char_dict_re
