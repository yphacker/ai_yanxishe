# coding=utf-8
# author=yphacker


import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def set_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def y_concatenate(y_true_list, y_pred_list, batch_y, logits):
    if y_true_list is None:
        y_true_list = batch_y.cpu().data.numpy()
        y_pred_list = logits.cpu().data.numpy()
    else:
        y_true_list = np.append(y_true_list, batch_y.cpu().data.numpy(), axis=0)
        y_pred_list = np.append(y_pred_list, logits.cpu().data.numpy(), axis=0)
    return y_true_list, y_pred_list


def get_score(y_true, y_pred):
    pearson_corr = pearsonr(y_pred, y_true)[0]
    spearman_corr = spearmanr(y_pred, y_true)[0]
    return (pearson_corr + spearman_corr) / 2


if __name__ == "__main__":
    pass
