# coding=utf-8
# author=yphacker

import gc
import os
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from importlib import import_module
from conf import config
from utils.data_utils import MyDataset
from utils.model_utils import FocalLoss, LabelSmoothingCrossEntropy
from utils.utils import set_seed, get_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_inputs(batch_x, batch_y=None):
    if model_name in ['bert']:
        batch_x = tuple(t.to(device) for t in batch_x)
        inputs = dict(input_ids=batch_x[0], attention_mask=batch_x[1])
        if model_name in ["bert"]:
            inputs['token_type_ids'] = batch_x[2]
        return inputs


def evaluate(model, val_iter, criterion):
    model.eval()
    data_len = 0
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch_x, batch_y in val_iter:
            batch_len = len(batch_y)
            data_len += batch_len
            inputs = get_inputs(batch_x, batch_y)
            batch_y = batch_y.to(device)
            logits = model(**inputs)
            preds = torch.argmax(logits, dim=1)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_len
            y_true_list += batch_y.cpu().data.numpy().tolist()
            y_pred_list += preds.cpu().data.numpy().tolist()
    return total_loss / data_len, get_score(np.array(y_true_list), np.array(y_pred_list))


def train(train_data, val_data, fold_idx=None):
    train_dataset = MyDataset(train_data, tokenizer)
    val_dataset = MyDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = model_file.Model().to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy(config.num_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))

    if os.path.isfile(model_save_path):
        print('加载之前的训练模型')
        try:
            model.load_state_dict(torch.load(model_save_path))
        except:
            print('加载失败')

    best_val_score = 0
    last_improved_epoch = 0
    adjust_lr_num = 0
    y_true_list = []
    y_pred_list = []
    for cur_epoch in range(config.epochs_num):
        start_time = int(time.time())
        model.train()
        print('epoch:{}, step:{}'.format(cur_epoch + 1, len(train_loader)))
        cur_step = 0
        for batch_x, batch_y in train_loader:
            inputs = get_inputs(batch_x, batch_y)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(**inputs)
            preds = torch.argmax(logits, dim=1)
            train_loss = criterion(logits, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            y_true_list += batch_y.cpu().data.numpy().tolist()
            y_pred_list += preds.cpu().data.numpy().tolist()
            if cur_step % config.train_print_step == 0:
                train_score = get_score(np.array(y_true_list), np.array(y_pred_list))
                msg = 'the current step: {0}/{1}, train score: {2:>6.2%}'
                print(msg.format(cur_step, len(train_loader), train_score))
                y_true_list = []
                y_pred_list = []
        val_loss, val_score = evaluate(model, val_loader, criterion)
        if val_score >= best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), model_save_path)
            improved_str = '*'
            last_improved_epoch = cur_epoch
        else:
            improved_str = ''
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val acc: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.epochs_num, val_loss, val_score,
                         end_time - start_time, improved_str))

        if cur_epoch - last_improved_epoch >= config.patience_epoch:
            if adjust_lr_num >= model_config.adjust_lr_num:
                print("No optimization for a long time, auto stopping...")
                break
            print("No optimization for a long time, adjust lr...")
            scheduler.step()
            last_improved_epoch = cur_epoch  # 加上，不然会连续更新的
            adjust_lr_num += 1
    del model
    gc.collect()

    if fold_idx is not None:
        model_score[fold_idx] = best_val_score


def eval():
    pass


def predict():
    model = model_file.Model().to(device)
    model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    test_df = pd.read_csv(config.test_path)

    test_dataset = MyDataset(test_df, tokenizer, 'test')
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size)

    preds_list = []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in tqdm(test_iter):
            inputs = get_inputs(batch_x)
            logits = model(**inputs)
            preds = torch.argmax(logits, dim=1)
            preds_list += [p.item() for p in preds]
    submission_df = pd.DataFrame({"id": range(len(preds_list)), "label": preds_list})
    submission_df['label'] = submission_df['label'].apply(lambda x: config.id2label[x])
    submission_df.to_csv('submission.csv', index=False, header=False)


def main(op):
    if op == 'train':
        train_df = pd.read_csv(config.train_path)
        # print(train_df['label'].value_counts())
        train_df['label'] = train_df['label'].apply(lambda x: config.label2id[x])
        # train_df = train_df[:1000]
        if args.mode == 1:
            x = train_df['text'].values
            y = train_df['label'].values
            skf = StratifiedKFold(n_splits=config.n_splits, random_state=0, shuffle=True)
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
                train(train_df.iloc[train_idx], train_df.iloc[val_idx], fold_idx)
            score = 0
            score_list = []
            for fold_idx in range(config.n_splits):
                score += model_score[fold_idx]
                score_list.append('{:.4f}'.format(model_score[fold_idx]))
            print('val score:{}, avg val score:{:.4f}'.format(','.join(score_list), score / config.n_splits))
        else:
            train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=0, shuffle=True)
            print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
            train(train_data, val_data)
    elif op == 'eval':
        pass
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    set_seed()
    parser = argparse.ArgumentParser(description='text classification by pytorch')
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=8, type=int, help="train epochs")
    parser.add_argument("-m", "--model", default='cnn', type=str, required=True,
                        help="choose a model: bert")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="train mode")
    # parser.add_argument('-em', '--embedding', default='pre_trained', type=str, help='random or pre_trained')
    args = parser.parse_args()

    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model

    model_file = import_module('models.{}'.format(model_name))
    model_config = import_module('conf.model_config_{}'.format(model_name))
    tokenizer = config.tokenizer_dict[model_name].from_pretrained(model_config.pretrain_model_path)

    model_score = dict()
    main(args.operation)
