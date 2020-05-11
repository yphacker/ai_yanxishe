# coding=utf-8
# author=yphacker

import argparse
import pandas as pd
from conf import config


def train():
    df = pd.read_csv(config.train_path)
    df.fillna('', inplace=True)
    df = pd.concat([df, df[df['label'] == 'spam']])
    df = df.sample(frac=1).reset_index(drop=True)
    x_train = df['text'].values.tolist()
    df['label'] = df['label'].apply(lambda x: config.labels_dict.index(x))
    y_train = df['label'].values.tolist()

    dev_sample_index = -1 * int(0.1 * float(len(y_train)))
    # 划分训练集和验证集
    # x_train, x_val = x_train[:dev_sample_index], x_train[dev_sample_index:]
    # y_train, y_val = y_train[:dev_sample_index], y_train[dev_sample_index:]
    x_train, x_val = x_train, x_train[dev_sample_index:]
    y_train, y_val = y_train, y_train[dev_sample_index:]
    print('train:{}, val:{}, all:{}'.format(len(y_train), len(y_val), df.shape[0]))

    model.train(x_train, y_train, x_val, y_val)


def eval():
    df = pd.read_csv(config.train_path, sep='\t', header=None, names=['label', 'text'])
    x_test = df['review'].values.tolist()

    preds = model.predict(x_test)
    df['pred_label'] = preds
    cols = ['pred_label', 'label', 'text']
    train = df.ix[:, cols]
    train.to_csv(config.train_check_path, index=False)


def predict():
    df = pd.read_csv(config.test_path)
    df.fillna('', inplace=True)
    x_test = df['text'].values.tolist()
    preds = model.predict(x_test)

    submission = pd.DataFrame({'id': range(len(preds)), 'pred': preds})
    submission['id'] = submission['id'] + 1
    submission.to_csv(config.model_submission_path, index=False, header=False)

    answer = pd.read_csv('../data/origin_data/answer.csv')
    answer['label'] = answer['label'].apply(lambda x: config.labels_dict.index(x))
    from sklearn.metrics import accuracy_score
    print(accuracy_score(answer['label'], submission.pred))


def main(op):
    if op == 'train':
        train()
    elif op == 'eval':
        eval()
    elif op == 'predict':
        predict()


def model_select(op='cnn'):
    model = None
    if op == 'cnn':
        from model.cnn_model import TextCNN
        from conf.cnn_model_config import model_submission_path
        model = TextCNN()
        config.model_submission_path = model_submission_path
    elif op == 'bert':
        from model.bert_model import BertModel
        from conf.bert_model_config import model_submission_path
        model = BertModel()
        config.model_submission_path = model_submission_path
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--EPOCHS", default=8, type=int, help="train epochs")
    parser.add_argument("-m", "--MODEL", default='cnn', type=str, help="model select")
    args = parser.parse_args()
    config.batch_size = args.BATCH
    config.epochs_num = args.EPOCHS
    model = model_select(args.MODEL)
    main(args.operation)
