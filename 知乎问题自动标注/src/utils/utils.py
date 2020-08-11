# coding=utf-8
# author=yphacker


import numpy as np
import torch


def set_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样




if __name__ == "__main__":
    pass
