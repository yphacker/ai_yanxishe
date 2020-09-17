# coding=utf-8
# author=yphacker


import numpy as np
import torch
from sklearn.metrics import accuracy_score


def set_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    pass
