# coding=utf-8
# author=yphacker

import torch
from torch.utils.data import Dataset
from conf import config


class MyDataset(Dataset):
    def __init__(self, df, tokenizer, mode="train"):
        super(MyDataset, ).__init__()
        self.tokenizer = tokenizer
        self.mode = mode
        self.x_data = []
        self.y_data = []
        for i, row in df.iterrows():
            x, y = self.row_to_tensor(self.tokenizer, row)
            self.x_data.append(x)
            self.y_data.append(y)

    def row_to_tensor(self, tokenizer, row):
        # source = row['source']
        # target = row['target']
        source = row['article'].replace('<Paragraph>', '。')
        source_inputs = tokenizer.encode_plus(source, max_length=1024, pad_to_max_length=True, return_tensors="pt")
        y_tensor = torch.tensor([0], dtype=torch.float32)
        if self.mode in ['train', 'val']:
            target = row['summarization']
            target_inputs = tokenizer.encode_plus(target, max_length=56, pad_to_max_length=True, return_tensors="pt")
            y_tensor = target_inputs['input_ids']

        # x_tensor = torch.tensor(source_inputs["input_ids"], dtype=torch.long), \
        #            torch.tensor(source_inputs['attention_mask'], dtype=torch.long)
        x_tensor = source_inputs["input_ids"].squeeze(), source_inputs['attention_mask'].squeeze()

        return x_tensor, y_tensor.squeeze()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y_data)
