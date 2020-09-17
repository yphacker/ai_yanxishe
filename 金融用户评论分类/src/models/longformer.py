# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
from transformers import LongformerModel

from conf import config
from conf import model_config_longformer as model_config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = LongformerModel.from_pretrained(model_config.pretrain_model_path, gradient_checkpointing=True)
        self.config = self.model.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
