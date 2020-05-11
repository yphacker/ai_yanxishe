# coding=utf-8
# author=yphacker

import os
import gc
import time
import argparse
from tqdm import tqdm
from importlib import import_module
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from conf import config
from conf import model_config_bert as model_config
from utils.data_utils import MyDataset
from utils.utils import y_concatenate, get_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_inputs(batch_x, batch_y=None):
    batch_x = tuple(t.to(device) for t in batch_x)
    inputs = dict(input_ids=batch_x[0], attention_mask=batch_x[1])
    if model_name in ["bert"]:
        inputs['token_type_ids'] = batch_x[2]
    return inputs


def evaluate(model, val_loader, criterion):
    model.eval()
    data_len = 0
    total_loss = 0
    y_true_list = None
    y_pred_list = None
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_len = len(batch_y)
            # batch_len = len(batch_y.size(0))
            data_len += batch_len
            inputs = get_inputs(batch_x, batch_y)
            batch_y = batch_y.to(device)
            logits = model(**inputs)
            loss = criterion(logits.view(-1), batch_y.view(-1))
            total_loss += loss.item()
            y_true_list, y_pred_list = y_concatenate(y_true_list, y_pred_list, batch_y, logits)
    y_pred_list = np.squeeze(y_pred_list)
    return total_loss / data_len, get_score(y_true_list, y_pred_list)


def train(train_data, val_data, fold_idx=None):
    train_dataset = MyDataset(train_data, tokenizer)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size)
    val_dataset = MyDataset(val_data, tokenizer)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size)

    model = model_file.Model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))

    best_val_score = 0
    last_improved_epoch = 0
    adjust_lr_num = 0
    y_true_list = None
    y_pred_list = None
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
            train_loss = criterion(logits.view(-1), batch_y.view(-1))
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            y_true_list, y_pred_list = y_concatenate(y_true_list, y_pred_list, batch_y, logits)
            if cur_step % config.train_print_step == 0:
                y_pred_list = np.squeeze(y_pred_list)
                train_score = get_score(y_true_list, y_pred_list)
                msg = 'the current step: {0}/{1}, train score: {2:>6.2%}'
                print(msg.format(cur_step, len(train_loader), train_score))
                y_true_list = None
                y_pred_list = None

        val_loss, val_score = evaluate(model, val_loader, criterion)
        if val_score >= best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), model_save_path)
            last_improved_epoch = cur_epoch
            improved_str = '*'
        else:
            improved_str = ''
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val score: {3:>6.2%}, cost: {4}s {5}'
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


def predict():
    model = model_file.Model().to(device)
    model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    test_data = pd.read_csv(config.test_path)
    test_dataset = MyDataset(test_data, tokenizer, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            inputs = get_inputs(batch_x)
            logits = model(**inputs)
            pred_list += [p.item() for p in logits]
    pred_list = np.squeeze(pred_list)
    submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
    submission.to_csv('submission.csv', index=False, header=False)


def main(op):
    if op == 'train':
        train_df = pd.read_csv(config.train_path)
        if args.mode == 1:
            x = train_df['text_a'].values
            y = train_df['score'].values
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
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=2, type=int, help="epochs num")
    parser.add_argument("-m", "--model", default='bert', type=str,
                        help="choose a model: bert, bart")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="train mode")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model
    model_file = import_module('models.{}'.format(model_name))
    model_config = import_module('conf.model_config_{}'.format(model_name))
    model_score = dict()
    tokenizer = config.tokenizer_dict[model_name].from_pretrained(model_config.pretrain_model_path)
    main(args.operation)
