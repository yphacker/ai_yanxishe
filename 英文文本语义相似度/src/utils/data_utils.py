# coding=utf-8
# author=yphacker

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from conf import config
from conf import model_config_bert


class MyDataset(Dataset):

    def __init__(self, df, tokenizer, mode='train'):
        self.mode = mode
        self.tokenizer = tokenizer
        self.x_data = []
        self.y_data = []
        for i, row in df.iterrows():
            x, y = self.row_to_tensor(self.tokenizer, row)
            self.x_data.append(x)
            self.y_data.append(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def row_to_tensor(self, tokenizer, row):
        inputs = tokenizer.encode_plus(row['text_a'], row['text_b'],
                                       max_length=config.max_seq_len, pad_to_max_length=True)
        y_tensor = torch.tensor(np.array(0), dtype=torch.float32)
        if self.mode in ['train', 'val']:
            y_tensor = torch.tensor(row['score'], dtype=torch.float32)
        x_tensor = torch.tensor(inputs["input_ids"], dtype=torch.long), \
                   torch.tensor(inputs['attention_mask'], dtype=torch.long), \
                   torch.tensor(inputs.get("token_type_ids", 0), dtype=torch.long)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y_data)


if __name__ == "__main__":
    pass
