# coding=utf-8
# author=yphacker


import torch
from torch.utils.data import Dataset
from conf import config


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
        text_a = row['query']
        # tokenizer.encode 自带截取功能
        inputs = tokenizer.encode_plus(text_a, max_length=config.max_seq_len, pad_to_max_length=True)
        y_tensor = torch.tensor(0, dtype=torch.long)
        if self.mode in ['train', 'val']:
            label = config.label2id[row['intent']]
            y_tensor = torch.tensor(label, dtype=torch.long)

        # 有些tokenizer.encode_plus返回不带token_type_ids
        x_tensor = torch.tensor(inputs["input_ids"], dtype=torch.long), \
                   torch.tensor(inputs['attention_mask'], dtype=torch.long), \
                   torch.tensor(inputs.get("token_type_ids", 0), dtype=torch.long)

        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y_data)


if __name__ == "__main__":
    pass
