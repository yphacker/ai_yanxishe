# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartForConditionalGeneration
from transformers.modeling_bart import BartClassificationHead
from conf import config
from conf import model_config_bart as model_config


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_config.pretrain_model_path)
        self.config = self.model.config

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
                lm_labels=None, generation_mode=False, **unused):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('bart-large')
            input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            lm_labels=lm_labels,
            generation_mode=generation_mode,
        )
        return outputs

    def generate(self, input_ids=None, max_length=None, min_length=None, do_sample=None, early_stopping=None,
                 num_beams=None,
                 temperature=None,
                 top_k=None,
                 top_p=None,
                 repetition_penalty=None,
                 bos_token_id=None,
                 pad_token_id=None,
                 eos_token_id=None,
                 length_penalty=None,
                 no_repeat_ngram_size=None,
                 num_return_sequences=None,
                 attention_mask=None,
                 decoder_start_token_id=None, ):
        return self.model.generate(input_ids=input_ids,
                                   max_length=max_length,
                                   min_length=min_length,
                                   do_sample=do_sample,
                                   early_stopping=early_stopping,
                                   num_beams=num_beams)
