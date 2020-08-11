# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from conf import config
from conf import model_config_bert as model_config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained(model_config.pretrain_model_path, output_hidden_states=True)
        self.model = BertModel.from_pretrained(model_config.pretrain_model_path, config=self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # return logits
        hidden_states = outputs[2][-4:]
        hidden = torch.stack(hidden_states, dim=-1).max(dim=-1)[0]  # [batch, seqlen, hidden_size]
        return self.classifier(hidden[:, 0, :])  # [batch, n_classes]
