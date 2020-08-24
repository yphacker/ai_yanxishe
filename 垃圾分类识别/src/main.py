# coding=utf-8
# author=yphacker

import gc
import os
import time
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
from ranger import Ranger
from torch.utils.data import DataLoader
from conf import config
from model.net import Net
from utils.data_utils import MyDataset
from utils.data_utils import train_transform, val_transform, test_transform
from utils.model_utils import accuracy, FocalLoss
from utils.utils import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, val_loader, criterion):
    # switch to evaluate mode
    model.eval()
    data_len = 0
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_len = len(batch_y)
            # batch_len = len(batch_y.size(0))
            data_len += batch_len
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            total_loss += loss.item()
            _acc = accuracy(probs, batch_y)
            total_acc += _acc[0].item() * batch_len

    return total_loss / data_len, total_acc / data_len


def train(train_data, val_data, fold_idx=None):
    train_data = MyDataset(train_data, train_transform)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    val_data = MyDataset(val_data, val_transform)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    model = Net(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    # optimizer = Ranger(model.parameters(), lr=1e-3, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))
    if os.path.isfile(model_save_path):
        print('加载之前的训练模型')
        model.load_state_dict(torch.load(model_save_path))

    best_val_score = 0
    best_val_score_cnt = 0
    last_improved_epoch = 0
    adjust_lr_num = 0
    for cur_epoch in range(config.epochs_num):
        start_time = int(time.time())
        model.train()
        print('epoch:{}, step:{}'.format(cur_epoch + 1, len(train_loader)))
        cur_step = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            probs = model(batch_x)

            train_loss = criterion(probs, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            if cur_step % config.train_print_step == 0:
                train_acc = accuracy(probs, batch_y)
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%}'
                print(msg.format(cur_step, len(train_loader), train_loss.item(), train_acc[0].item()))
        val_loss, val_score = evaluate(model, val_loader, criterion)
        if val_score >= best_val_score:
            if val_score == best_val_score:
                best_val_score_cnt += 1
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
        if cur_epoch - last_improved_epoch >= config.patience_epoch or best_val_score_cnt >= 3:
            if adjust_lr_num >= config.adjust_lr_num:
                print("No optimization for a long time, auto stopping...")
                break
            print("No optimization for a long time, adjust lr...")
            # scheduler.step()
            last_improved_epoch = cur_epoch  # 加上，不然会连续更新的
            adjust_lr_num += 1
            best_val_score_cnt = 0
        # scheduler.step()
    del model
    gc.collect()

    if fold_idx is not None:
        model_score[fold_idx] = best_val_score


def predict():
    model = Net(model_name).to(device)
    model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    model.load_state_dict(torch.load(model_save_path))

    data_len = len(os.listdir(config.image_test_path))
    test_path_list = ['{}/{}.jpg'.format(config.image_test_path, x) for x in range(0, data_len)]
    test_data = np.array(test_path_list)
    test_dataset = MyDataset(test_data, test_transform, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            batch_x = batch_x.to(device)
            # compute output
            probs = model(batch_x)
            preds = torch.argmax(probs, dim=1)
            pred_list += [p.item() for p in preds]

    submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
    submission['label'] = submission['label'].apply(lambda x: config.id2label[x])
    submission.to_csv('submission.csv', index=False, header=False)


def main(op):
    if op == 'train':
        train_df = pd.read_csv('../data/train.csv')
        print(train_df['label'].value_counts())
        train_df['filename'] = train_df['filename'].apply(lambda x: '../data/train/{0}'.format(x))
        train_df['label'] = train_df['label'].apply(lambda x: config.label2id[x])
        if mode == 1:
            n_splits = 5
            x = train_df['filename'].values
            y = train_df['label'].values
            skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
                train(train_df.iloc[train_idx], train_df.iloc[val_idx], fold_idx)
            score = 0
            score_list = []
            for fold_idx in range(config.n_splits):
                score += model_score[fold_idx]
                score_list.append('{:.4f}'.format(model_score[fold_idx]))
            print('val score:{}, avg val score:{:.4f}'.format(','.join(score_list), score / config.n_splits))
        else:
            train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
            print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
            train(train_data, val_data)
    elif op == 'eval':
        pass
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=16, type=int, help="train epochs")
    parser.add_argument("-m", "--model_name", default='resnet', type=str, help="model select")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="train mode")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model_name
    mode = args.mode

    set_seed()
    model_score = dict()
    main(args.operation)
