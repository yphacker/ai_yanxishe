# coding=utf-8
# author=yphacker

import os
import json
import numpy as np
import pandas as pd
from conf import config


def prepare_model_settings():
    time_shift_ms = 100.0
    sample_rate = 16000
    clip_duration_ms = 1000
    window_size_ms = 30.0
    window_stride_ms = 10.0
    dct_coefficient_count = 40
    num_labels = 8
    """Calculates common settings needed for all models.

    Args:
      num_labels: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'num_labels': num_labels,
        'sample_rate': sample_rate,
    }


def batch_iter(x, y, batch_size=config.batch_size):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


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
    f = pd.read_csv(filein, usecols=['label'])
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
    label2id = {"Healthy": 0, "COPD": 1, "LRTI": 2, "URTI": 3, "Bronchiectasis": 4,
                 "Pneumonia": 5, "Bronchiolitis": 6, "Asthma": 7}
    id2label = {v:k for k,v in label2id.items()}
    return label2id, id2label
