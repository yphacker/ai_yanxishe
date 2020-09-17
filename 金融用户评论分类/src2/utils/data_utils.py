# coding=utf-8
# author=yphacker

import re
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

    def solve(self, features):
        # print([len(feature['input_ids']) for feature in features],
        #       [len(feature['attention_mask']) for feature in features],
        #       [len(feature['token_type_ids']) for feature in features])
        return torch.tensor([feature['input_ids'] for feature in features], dtype=torch.long), \
               torch.tensor([feature['attention_mask'] for feature in features], dtype=torch.long), \
               torch.tensor([feature['token_type_ids'] for feature in features], dtype=torch.long)

    def row_to_tensor(self, tokenizer, row):
        # text = row['text']
        text = row['text']
        text = re.sub("(\\W)+", " ", text)
        text = tokenizer.tokenize(text)

        skip_len = len(text) / config.split_num
        features = []
        for i in range(config.split_num):
            text_tmp = text[int(i * skip_len):int((i + 1) * skip_len)]
            inputs = tokenizer.encode_plus(text_tmp, max_length=config.max_seq_len, pad_to_max_length=True)
            features.append(dict(input_ids=inputs["input_ids"],
                                 attention_mask=inputs['attention_mask'],
                                 token_type_ids=inputs.get("token_type_ids", 0)))
            # context_tokens_choice = text[int(i * skip_len):int((i + 1) * skip_len)]
            #
            # tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"]
            # input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # input_mask = [1] * len(input_ids)
            # segment_ids = [0] * (len(context_tokens_choice) + 2)
            #
            # padding_length = config.max_seq_len - len(input_ids)
            # input_ids += ([0] * padding_length)
            # input_mask += ([0] * padding_length)
            # segment_ids += ([0] * padding_length)
            # features.append(dict(input_ids=input_ids,
            #                      attention_mask=input_mask,
            #                      token_type_ids=segment_ids))

        x_tensor = self.solve(features)
        y_tensor = torch.tensor(0, dtype=torch.long)
        if self.mode in ['train', 'val']:
            y_tensor = torch.tensor(row['label'], dtype=torch.long)

        return x_tensor, y_tensor

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
