# coding=utf-8
# author=yphacker

import gc
import os
import cv2
import time
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
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
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_len = len(batch_y)
            # batch_len = len(batch_y.size(0))
            data_len += batch_len
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            total_loss += loss.item()

    return total_loss / data_len


def train(train_data, val_data, fold_idx=None):
    train_data = MyDataset(train_data, train_transform)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    val_data = MyDataset(val_data, val_transform)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    model = Net(model_name).to(device)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    optimizer = Ranger(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))
    # if os.path.isfile(model_save_path):
    #     print('加载之前的训练模型')
    #     model.load_state_dict(torch.load(model_save_path))

    best_val_loss = 1000
    best_val_loss_cnt = 0
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
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}'
                print(msg.format(cur_step, len(train_loader), train_loss.item()))
        val_loss = evaluate(model, val_loader, criterion)
        if val_loss <= best_val_loss:
            if val_loss == best_val_loss:
                best_val_loss_cnt += 1
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            improved_str = '*'
            last_improved_epoch = cur_epoch
        else:
            improved_str = ''
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, cost: {3}s {4}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.epochs_num, val_loss,
                         end_time - start_time, improved_str))
        if cur_epoch - last_improved_epoch >= config.patience_epoch or best_val_loss_cnt >= 3:
            if adjust_lr_num >= config.adjust_lr_num:
                print("No optimization for a long time, auto stopping...")
                break
            print("No optimization for a long time, adjust lr...")
            # scheduler.step()
            last_improved_epoch = cur_epoch  # 加上，不然会连续更新的
            adjust_lr_num += 1
            best_val_score_cnt = 0
        scheduler.step()
    del model
    gc.collect()

    if fold_idx is not None:
        model_score[fold_idx] = best_val_loss


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
            logits = model(batch_x)
            output = logits.data.cpu().numpy()
            pred_list.extend(output)

    submission_df = pd.DataFrame(pred_list)
    submission_df.columns = ['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
                             'mouth_x', 'mouth_y', 'left_ear1_x', 'left_ear1_y', 'left_ear2_x', 'left_ear2_y',
                             'left_ear3_x', 'left_ear3_y', 'right_ear1_x', 'right_ear1_y',
                             'right_ear2_x', 'right_ear2_y', 'right_ear3_x', 'right_ear3_y']
    submission_df = submission_df.reset_index()

    img_size = []
    for idx in (range(data_len)):
        img_size.append(cv2.imread('{}/{}.jpg'.format(config.image_test_path, idx)).shape[:2])

    img_size = np.vstack(img_size)
    submission_df['height'] = img_size[:, 0]
    submission_df['width'] = img_size[:, 1]

    for col in submission_df.columns:
        if '_x' in col:
            submission_df[col] *= submission_df['width']
        elif '_y' in col:
            submission_df[col] *= submission_df['height']

    submission_df.astype(int).iloc[:, :-2].to_csv('submission.csv', index=None, header=None)


def main(op):
    if op == 'train':
        train_df = pd.read_csv('../data/train.csv')
        train_df['filename'] = train_df['filename'].apply(lambda x: '../data/train/{0}.jpg'.format(x))
        random_state = 0
        if mode == 1:
            n_splits = 5
            x = train_df['filename'].values
            y = train_df['label'].values
            skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
                train(train_df.iloc[train_idx], train_df.iloc[val_idx], fold_idx)
            score = 0
            score_list = []
            for fold_idx in range(config.n_splits):
                score += model_score[fold_idx]
                score_list.append('{:.4f}'.format(model_score[fold_idx]))
            print('val score:{}, avg val score:{:.4f}'.format(','.join(score_list), score / config.n_splits))
        else:
            train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=random_state, shuffle=True)
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
