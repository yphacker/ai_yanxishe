# coding=utf-8
# author=yphacker

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from conf import config


def train():
    train_df = pd.read_json('../data/train.json')
    train_df = train_df[:50]
    train_df['is_spoiler'] = train_df['is_spoiler'].apply(lambda x: 1 if x else 0)
    train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
    x_train, x_val = train_data['review_text'].values.tolist(), val_data['review_text'].values.tolist()
    y_train, y_val = train_data['is_spoiler'].values.tolist(), val_data['is_spoiler'].values.tolist()

    model.train(x_train, y_train, x_val, y_val)


def eval():
    pass


def predict():
    test_df = pd.read_json('../data/test.json')
    x_test = test_df['review_text'].values.tolist()
    preds = model.predict(x_test)

    submission = pd.DataFrame({'id': range(len(preds)), 'pred': preds})
    submission['id'] = submission['id']
    submission.to_csv(config.model_submission_path, index=False, header=False)


def main(op):
    if op == 'train':
        train()
    elif op == 'eval':
        eval()
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--EPOCHS", default=8, type=int, help="train epochs")
    parser.add_argument("-m", "--MODEL", default='cnn', type=str, help="model select")
    args = parser.parse_args()
    config.batch_size = args.BATCH
    config.epochs_num = args.EPOCHS
    from model.bert_model import BertModel
    from conf.bert_model_config import model_submission_path

    model = BertModel()
    config.model_submission_path = model_submission_path
    main(args.operation)
