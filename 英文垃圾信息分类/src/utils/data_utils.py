# coding=utf-8
# author=yphacker

from conf import config



def load_label_dict():
    label2id = dict()
    id2label = dict()
    for k, v in enumerate(config.labels):
        label2id[v] = k
        id2label[k] = v
    return label2id, id2label
