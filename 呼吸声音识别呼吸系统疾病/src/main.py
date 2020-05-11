# coding=utf-8
# author=yphacker


import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from conf import config
from utils.utils import load_label_dict


def count_label(t):
    cnt = [0] * 8
    for tt in t:
        for i in range(8):
            if tt[i] == 1:
                cnt[i] += 1
    print(cnt)


def train():
    x_numpy = np.load('../data/x_train.npy')
    x_numpy = x_numpy.tolist()
    y_numpy = np.load('../data/y_train.npy')
    y_numpy = y_numpy.tolist()
    print(count_label(y_numpy))
    x_train, x_val, y_train, y_val = train_test_split(x_numpy, y_numpy, test_size=0.1, random_state=0)
    print(count_label(y_train))
    print(count_label(y_val))
    print('train:{}, val:{}, all:{}'.format(len(y_train), len(y_val), len(x_numpy)))

    model.train(x_train, y_train, x_val, y_val)


def predict():
    label2id, id2label = load_label_dict()
    test_df = pd.read_csv(config.test_path)
    x_test = np.load('../data/x_test.npy')
    preds = model.predict(x_test)
    print(preds)
    pred_df = pd.DataFrame({'patient_id': test_df['patient_id'],
                            'diagnosis': [id2label[var] for var in preds]})
    pred_df.to_csv('../data/pred.csv', index=None)


def main(op):
    if op == 'train':
        train()
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=16, type=int, help="train epochs")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    from models.cnn import Model

    model = Model()
    main(args.operation)
