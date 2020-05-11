# coding=utf-8
# author=yphacker


import os
import pandas as pd
from matplotlib import cm
from tqdm import tqdm
import pylab
import librosa
from librosa import display
import numpy as np
from conf import config


def create_image(source_filepath, destination_filepath):
    y, sr = librosa.load(source_filepath, sr=22050)  # Use the default sampling rate of 22,050 Hz

    # Pre-emphasis filter
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Compute spectrogram
    M = librosa.feature.melspectrogram(y,
                                       sr,
                                       fmax=sr / 2,  # Maximum frequency to be used on the on the MEL scale
                                       n_fft=2048,
                                       hop_length=512,
                                       n_mels=96,  # As per the Google Large-scale audio CNN paper
                                       power=2)  # Power = 2 refers to squared amplitude
    # librosa.feature.melspectrogram(y=clip, sr=sample_rate,n_fft=2048, hop_length=512)
    # librosa.feature.mfcc(y=clip, sr=sr, dct_type=2)
    # librosa.feature.chroma_stft(S=S, sr=sr)
    # Power in DB
    log_power = librosa.power_to_db(M, ref=np.max)  # Covert to dB (log) scale

    # Plotting the spectrogram and save as JPG without axes (just the image)
    pylab.figure(figsize=(5, 5))  # was 14, 5
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    librosa.display.specshow(log_power, cmap=cm.jet)
    pylab.savefig(destination_filepath, bbox_inches=None, pad_inches=0)
    pylab.close()


def get_image_feature(origin_path, new_path):
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    num = 0
    for filename in tqdm(os.listdir(origin_path)):
        if 'train' in new_path:
            label = filename.split('.')[0].split('-')[-1]
            new_filename = '{}_{}.jpg'.format(num, label)
        else:
            new_filename = '{}.jpg'.format(filename.split('.')[0])
        origin_filepath = '{}/{}'.format(origin_path, filename)
        new_filepath = '{}/{}'.format(new_path, new_filename)
        create_image(origin_filepath, new_filepath)
        num += 1


def get_data():
    filenames = os.listdir('../data/train')
    train_df = pd.DataFrame({'filename': filenames})

    train_df['label'] = train_df['filename'].apply(lambda x: int(x.split('.')[0].split('_')[-1]))
    train_df.to_csv(config.train_path, index=None)


if __name__ == '__main__':
    # get_image_feature('../data/origin/train', '../data/train')
    get_image_feature('../data/origin/test', '../data/test')
    # get_data()
