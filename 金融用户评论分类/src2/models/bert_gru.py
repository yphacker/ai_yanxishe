# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
from transformers import BertModel
from conf import config
from conf import model_config_bert_gru as model_config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = BertModel.from_pretrained(model_config.pretrain_model_path)
        self.config = self.model.config
        # self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # self.classifier = nn.Linear(self.config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(model_config.lstm_dropout)
        self.classifier = nn.Linear(model_config.lstm_hidden_size * 2, config.num_labels)
        self.gru = []
        for i in range(model_config.lstm_layers):
            self.gru.append(
                nn.GRU(self.config.hidden_size, model_config.lstm_hidden_size,
                       num_layers=model_config.lstm_layers, bidirectional=True, batch_first=True).cuda())
        self.gru = nn.ModuleList(self.gru)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None):
        tmp_input_ids = input_ids.view(-1, input_ids.size(-1))
        tmp_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        tmp_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        outputs = self.model(
            tmp_input_ids,
            attention_mask=tmp_attention_mask,
            token_type_ids=tmp_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # return logits
        pooled_output = outputs[1]
        output = pooled_output.reshape(input_ids.size(0), input_ids.size(1), -1).contiguous()

        for gru in self.gru:
            gru.flatten_parameters()
            output, hidden = gru(output)
            output = self.dropout(output)

        hidden = hidden.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()
        logits = self.classifier(hidden)
        return logits
