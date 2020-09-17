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
            x, y = self.row_to_tensor(self.tokenizer, row, i)
            self.x_data.extend(x)
            self.y_data.extend(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def row_to_tensor(self, tokenizer, row, index):
        # text = row['text']
        text = tokenizer.tokenize(row['text'])

        y_tensor = torch.tensor(0, dtype=torch.long)
        if self.mode in ['train', 'val']:
            y_tensor = torch.tensor(row['label'], dtype=torch.long)

        # skip_len = config.max_seq_len - 50
        # skip_len2 = config.max_seq_len - 10
        # split_num = int((len(text) - 1) / config.max_seq_len) + 1
        skip_len = config.max_seq_len
        split_num = int((len(text) - 1) / config.max_seq_len) + 1
        x_tensors = []
        y_tensors = []
        for i in range(split_num):
            # text_tmp = text[i * skip_len:i * skip_len + skip_len2]
            text_tmp = text[i * skip_len:i * skip_len + skip_len]
            inputs = tokenizer.encode_plus(text_tmp, max_length=config.max_seq_len, pad_to_max_length=True)
            x_tensors.append((torch.tensor(inputs["input_ids"], dtype=torch.long),
                              torch.tensor(inputs['attention_mask'], dtype=torch.long),
                              torch.tensor(inputs.get("token_type_ids", 0), dtype=torch.long),
                              index))
            y_tensors.append(y_tensor)

        return x_tensors, y_tensors

    def __len__(self):
        return len(self.y_data)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


if __name__ == "__main__":
    pass
