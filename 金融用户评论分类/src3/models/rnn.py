# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import config


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.lstm = nn.LSTM(768, 100, batch_first=True)
#         self.fc1 = nn.Linear(100, 30)
#         self.fc2 = nn.Linear(30, config.num_labels)
#
#     def forward(self, inputs):
#         output, (hidden, cell) = self.lstm(inputs)  # 1 * batch_size * 768
#         hidden = hidden.squeeze(0)  # batch_size * 768
#         hidden = F.relu(self.fc1(hidden))  # batch_size * 30
#         # hidden = F.softmax(self.fc2(hidden), dim=1)  # batch_size * 10
#         # return hidden
#         mask = (inputs != -99.).view(-1)
#         hidden = hidden[mask]
#         logits = self.fc2(hidden)
#         return logits


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
#         self.dropout = nn.Dropout(0.5)
#         self.classifier = nn.Linear(128 * 2, config.num_labels)
#
#     def forward(self, inputs):
#         output, (hidden, cell) = self.lstm(inputs)  # 1 * batch_size * 768
#         hidden = hidden.squeeze(0)  # batch_size * 768
#         hidden = self.dropout(hidden)
#         hidden = hidden.permute(1, 0, 2).reshape(inputs.size(0), -1).contiguous()
#         logits = self.classifier(hidden)
#         return logits

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.rnn = nn.GRU(768, 128, bidirectional=True, batch_first=True)
        self.rnn = nn.GRU(1024, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.8)
        self.classifier = nn.Linear(128*2, config.num_labels)

    def forward(self, inputs):
        output, hidden = self.rnn(inputs)
        hidden = hidden.squeeze(0)  # batch_size * 768
        hidden = self.dropout(hidden)
        hidden = hidden.permute(1, 0, 2).reshape(inputs.size(0), -1).contiguous()
        logits = self.classifier(hidden)
        return logits

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.gru = nn.GRU(768, 128,
#                           num_layers=1, bidirectional=True, batch_first=True)
#         self.dropout = nn.Dropout(0.5)
#         self.classifier = nn.Linear(128 * 4, config.num_labels)
#
#     def forward(self, inputs):
#         h_gru, hh_gru = self.gru(inputs)
#
#         avg_pool = torch.mean(h_gru, 1)
#         max_pool, _ = torch.max(h_gru, 1)
#
#         conc = torch.cat(
#             (avg_pool, max_pool), 1
#         )
#
#         x = self.dropout(conc)
#         logits = self.classifier(x)
#         return logits
