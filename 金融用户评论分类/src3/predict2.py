# coding=utf-8
# author=yphacker

import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from importlib import import_module
from conf import config
from utils.data_utils2 import MyDataset, collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_inputs(batch_x, batch_y=None):
    inputs = dict()
    inputs['inputs'] = batch_x.to(device)
    return inputs


def predict(model, test_loader):
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            inputs = get_inputs(batch_x)
            # compute output
            logits = model(**inputs)
            probs = torch.softmax(logits, dim=1)
            pred_list += [_.cpu().data.numpy() for _ in probs]
    return pred_list


def model_predict(model_name):
    test_dataset = MyDataset(test_df, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    preds_dict = dict()
    for fold_idx in range(config.n_splits):
        model = model_file.Model().to(device)
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))
        model.load_state_dict(torch.load(model_save_path))
        pred_list = predict(model, test_loader)
        submission = pd.DataFrame(pred_list)
        submission.to_csv('{}/{}_fold{}_submission.csv'
                          .format(config.submission_path, model_name, fold_idx), index=False, header=False)
        preds_dict['{}_{}'.format(model_name, fold_idx)] = pred_list
    pred_list = get_pred_list(preds_dict)

    submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
    submission.to_csv('submission.csv', index=False, header=False)


def file2submission():
    preds_dict = dict()
    for model_name in model_name_list:
        for fold_idx in range(5):
            df = pd.read_csv('{}/{}_fold{}_submission.csv'
                             .format(config.submission_path, model_name, fold_idx), header=None)
            preds_dict['{}_{}'.format(model_name, fold_idx)] = df.values
    pred_list = get_pred_list(preds_dict)

    submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
    submission.to_csv('submission.csv', index=False, header=False)


def get_pred_list(preds_dict):
    pred_list = []
    if mode == 1:
        for i in range(data_len):
            preds = []
            for model_name in model_name_list:
                for fold_idx in range(config.n_splits):
                    prob = preds_dict['{}_{}'.format(model_name, fold_idx)][i]
                    pred = np.argmax(prob)
                    preds.append(pred)
            # pred_set = set([x for x in preds])
            pred_list.append(max(preds, key=preds.count))
    else:
        for i in range(data_len):
            prob = None
            for model_name in model_name_list:
                for fold_idx in range(config.n_splits):
                    if prob is None:
                        prob = preds_dict['{}_{}'.format(model_name, fold_idx)][i] * ratio_dict[model_name]
                    else:
                        prob += preds_dict['{}_{}'.format(model_name, fold_idx)][i] * ratio_dict[model_name]
            pred_list.append(np.argmax(prob))
    return pred_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("-m", "--model_names", default='cnn', type=str, help="cnn")
    parser.add_argument("-type", "--pred_type", default='model', type=str, help="pred type")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="1:投票融合，2:加权融合")
    parser.add_argument("-r", "--ratios", default='1', type=str, help="融合比例")
    parser.add_argument("-files", "--files", default='1', type=str, help="融合比例")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    model_name_list = args.model_names.split('+')
    ratio_dict = dict()
    ratios = args.ratios
    ratio_list = args.ratios.split(',')
    for i, ratio in enumerate(ratio_list):
        ratio_dict[model_name_list[i]] = int(ratio)
    mode = args.mode

    test_array = np.load('../data/new_test.npz')
    test_df = pd.DataFrame({'emb': test_array['emb'], 'label': test_array['label']})
    data_len = test_df.shape[0]

    if args.pred_type == 'model':
        model_name = args.model_names
        model_file = import_module('models.{}'.format(model_name))
        model_config = import_module('conf.model_config_{}'.format(model_name))
        model_predict(model_name)
    elif args.pred_type == 'file':
        file2submission()
