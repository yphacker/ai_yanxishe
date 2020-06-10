# coding=utf-8
# author=yphacker

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from conf import config


def set_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def y_concatenate(y_true_list, y_pred_list, batch_y, logits):
    if y_true_list is None:
        y_true_list = y_true_list, batch_y.cpu().data.numpy()
        y_pred_list = y_pred_list, logits.cpu().data.numpy()
    else:
        y_true_list = np.append(y_true_list, batch_y.cpu().data.numpy(), axis=0)
        y_pred_list = np.append(y_pred_list, logits.cpu().data.numpy(), axis=0)
    return y_true_list, y_pred_list


def get_score(y_true, y_pred):
    score = 0
    for i, label in enumerate(config.label_columns):
        try:
            # print('{} roc_auc: {}'.format(label, roc_auc_score(y_true[:, i], y_pred[:, i])))
            score += roc_auc_score(y_true[:, i], y_pred[:, i])
        except:
            continue
    return score / len(config.label_columns)


if __name__ == "__main__":
    pass
