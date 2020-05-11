# coding=utf-8
# author=yphacker

import os
import re
import soundfile
import numpy as np
import pandas as pd
from pydub import AudioSegment
import tensorflow as tf
from conf import config
from utils.utils import prepare_model_settings, load_label_dict
from audio_processor import AudioProcessor

audio_processor = AudioProcessor()
model_settings = prepare_model_settings()
audio_processor.prepare_processing_graph(model_settings)
label2id, id2label = load_label_dict()


def deal_wav(path):
    data, samplerate = soundfile.read(path)
    soundfile.write('tmp.wav', data, samplerate, subtype='PCM_16')

    with tf.Session() as sess:
        data = audio_processor.get_data('tmp.wav', model_settings, 0, sess)
    return np.squeeze(data, axis=0)


def deal_y(diagnosis):
    y = np.zeros((config.num_labels,))
    y[label2id[diagnosis]] = 1
    return y


def get_train_data():
    # x_train_df = pd.read_csv('../data/origin/train_demographic_info.csv',
    #                          names=['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height'])
    y_train_df = pd.read_csv('../data/origin/train_patient_diagnosis.csv', names=['patient_id', 'diagnosis'])
    # train_df = pd.merge(x_train_df, y_train_df, how='inner', on='patient_id')
    # train_df['audio_and_txt_files_path'] = train_df['patient_id'].apply(lambda x: '../data/train/{}'.format(x))
    # train_df.to_csv('../data/train.csv', index=None)
    # df = pd.concat([df, df[df['diagnosis'] != 'COPD']])

    origin_train_path = '../data/origin/train'
    x_train = []
    y_train = []
    for filename in os.listdir(origin_train_path):
        if 'wav' in filename:
            path = '{}/{}'.format(origin_train_path, filename)
            x = deal_wav(path)
            patient_id = filename.split('_')[0]
            diagnosis = y_train_df[y_train_df['patient_id'] == int(patient_id)]['diagnosis'].to_list()[0]
            y = deal_y(diagnosis)
            # [29, 655, 2, 16, 16, 37, 13, 1]
            # label_cnt = [29, 655, 2, 16, 16, 37, 13, 1]
            # label_sum = sum([29, 655, 2, 16, 16, 37, 13, 1])
            # label_id = label2id[diagnosis]
            # for i in range(int(label_sum / label_cnt[label_id])):
            #     x_train.append(x)
            #     y_train.append(y)
            x_train.append(x)
            y_train.append(y)

    np.save('../data/x_train.npy', np.asarray(x_train))
    np.save('../data/y_train.npy', np.asarray(y_train))
    os.system("rm {}".format('./tmp.wav'))


def get_test_data():
    # test_df = pd.read_csv('../data/origin/test_demographic_info.csv',
    #                       names=['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height'])
    # test_df['audio_and_txt_files_path'] = test_df['patient_id'].apply(lambda x: '../data/train/{}'.format(x))

    origin_test_path = '../data/origin/test'
    x_test = []
    patient_id_list = []
    for filename in os.listdir(origin_test_path):
        if 'wav' in filename:
            path = '{}/{}'.format(origin_test_path, filename)
            x_test.append(deal_wav(path))
            patient_id = filename.split('_')[0]
            patient_id_list.append(patient_id)

    np.save('../data/x_test.npy', np.asarray(x_test))
    test_df = pd.DataFrame({'patient_id': patient_id_list})
    test_df.to_csv('../data/test.csv', index=None)
    os.system("rm {}".format('./tmp.wav'))


if __name__ == '__main__':
    get_train_data()
    # get_test_data()
