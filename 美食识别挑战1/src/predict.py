# coding=utf-8
# author=yphacker

import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from conf import config
from model.net import Net
from utils.data_utils import MyDataset
from utils.data_utils import test_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model):
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
            # preds = torch.argmax(probs, dim=1)
            # pred_list += [p.item() for p in preds]
            pred_list.extend(probs.cpu().numpy())
    return pred_list


def multi_model_predict():
    preds_dict = dict()
    for model_name in model_name_list:
        for fold_idx in range(5):
            model = Net(model_name).to(device)
            model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))
            model.load_state_dict(torch.load(model_save_path))
            pred_list = predict(model)
            submission = pd.DataFrame(pred_list)
            # submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
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
                for fold_idx in range(5):
                    prob = preds_dict['{}_{}'.format(model_name, fold_idx)][i]
                    pred = np.argmax(prob)
                    preds.append(pred)
            # pred_set = set([x for x in preds])
            pred_list.append(max(preds, key=preds.count))
    else:
        for i in range(data_len):
            prob = None
            for model_name in model_name_list:
                for fold_idx in range(5):
                    if prob is None:
                        prob = preds_dict['{}_{}'.format(model_name, fold_idx)][i] * ratio_dict[model_name]
                    else:
                        prob += preds_dict['{}_{}'.format(model_name, fold_idx)][i] * ratio_dict[model_name]
            pred_list.append(np.argmax(prob))
    return pred_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("-m", "--model_names", default='resnet', type=str, help="model select")
    parser.add_argument("-type", "--pred_type", default='model', type=str, help="model select")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="1:投票融合，2:相加融合")
    parser.add_argument("-r", "--ratios", default='1', type=str, help="融合比例")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    model_name_list = args.model_names.split('+')
    ratio_dict = dict()
    ratios = args.ratios
    ratio_list = args.ratios.split(',')
    for i, ratio in enumerate(ratio_list):
        ratio_dict[model_name_list[i]] = int(ratio)
    mode = args.mode
    data_len = len(os.listdir(config.image_test_path))
    if args.pred_type == 'model':
        multi_model_predict()
    elif args.pred_type == 'file':
        file2submission()
