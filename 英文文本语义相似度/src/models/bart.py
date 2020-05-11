# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
from transformers import BartModel
from transformers.modeling_bart import BartClassificationHead
from conf import config
from conf import model_config_bart as model_config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = BartModel.from_pretrained(model_config.pretrain_model_path)
        self.config = self.model.config
        self.classification_head = BartClassificationHead(
            self.config.d_model, self.config.d_model, config.num_labels, self.config.classif_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
        )
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)
        return logits
