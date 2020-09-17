# coding=utf-8
# author=yphacker


import torch
from torch.utils.data import Dataset
from conf import config


class MyDataset(Dataset):

    def __init__(self, df, mode='train'):
        self.mode = mode
        self.tokenizer = None
        self.x_data = []
        self.y_data = []
        for i, row in df.iterrows():
            x, y = self.row_to_tensor(row)
            self.x_data.append(x)
            self.y_data.append(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def row_to_tensor(self, row):
        x_tensor = torch.tensor(row['emb'])
        y_tensor = torch.tensor(0, dtype=torch.long)
        if self.mode in ['train', 'val']:
            y_tensor = torch.tensor(row['label'], dtype=torch.long)

        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y_data)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    x_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]
    max_len = len(max(x_data, key=len))
    import numpy as np
    # tmp = np.full((len(batch), max_len, 768), -99.)
    tmp = np.full((len(batch), max_len, 1024), -99.)
    for i in range(len(batch)):
        tmp[i, 0:len(x_data[i]), :] = x_data[i]

    x_tensor = torch.tensor(tmp, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    return x_tensor, y_tensor


if __name__ == "__main__":
    pass
