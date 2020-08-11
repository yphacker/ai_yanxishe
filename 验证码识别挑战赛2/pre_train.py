# coding=utf-8
# author=yphacker

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def main():
    train_df = pd.read_csv('../data/train.csv')
    train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=0, shuffle=True)
    print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
    train_data.to_csv('lib/dataset/txt/train.csv', index=False, header=False)
    val_data.to_csv('lib/dataset/txt/val.csv', index=False, header=False)


def get_alphabets():
    infile = open('../data/hanzi.txt', 'r')
    items = []
    for line in infile:
        item = line.strip()
        items.append(item)
    print(''.join(items))
    infile.close()


if __name__ == '__main__':
    main()
    # get_alphabets()
