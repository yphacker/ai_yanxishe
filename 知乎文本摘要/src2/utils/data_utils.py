# coding=utf-8
# author=yphacker

from torch.autograd import Variable
import numpy as np
import torch
from torch.utils.data import Dataset
from data_util import config


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

        self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
        # Store the in-article OOVs themselves
        self.art_oovs = [ex.article_oovs for ex in example_list]
        # Store the version of the enc_batch that uses the article OOV ids
        self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def row_to_tensor(self, tokenizer, row):
        source = row['article']
        target = row['summarization']
        # Use the <s> and </s> tags in abstract to get a list of sentences.
        abstract_sentences = [sent.strip() for sent in abstract2sents(target)]
        example = Example(source, abstract_sentences, self._vocab)  # Process into an Example.

        # enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
        # enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
        # enc_lens = batch.enc_lens
        # extra_zeros = None
        # enc_batch_extend_vocab = None
        #
        # if config.pointer_gen:
        #     enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        #     # max_art_oovs is the max over all the article oov list in the batch
        #     if batch.max_art_oovs > 0:
        #         extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
        input_ids = example.enc_input
        input_len = len(input_ids)
        input_mask = [1] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = config.max_seq_len - len(input_ids)

        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length

        extra_zeros = [0] *  len(10000)
        c_t_1 = [0] * (2 * config.hidden_dim)

        x_tensor = input_ids, input_mask, input_len, example.enc_input_extend_vocab, \
                   extra_zeros, c_t_1, example.coverage

        coverage = [0] *1000000


        dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
        dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
        dec_lens = batch.dec_lens
        max_dec_len = np.max(dec_lens)
        dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

        target_batch = Variable(torch.from_numpy(batch.target_batch)).long()



        y_tensor = dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

        return x_tensor, example

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y_data)

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    x_data = [item[0] for item in batch]
    y_data = [item[1] for item in batch]

    input_lens = [x[3] for x in x_data]
    max_len = max(input_lens)

    input_ids = [x[0][:max_len] for x in x_data]
    input_mask = [x[1][:max_len] for x in x_data]
    segment_ids = [x[2][:max_len] for x in x_data]
    start_ids = [x[0][:max_len] for x in y_data]
    end_ids = [x[1][:max_len] for x in y_data]

    x_tensor = torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(input_mask, dtype=torch.long), \
               torch.tensor(segment_ids, dtype=torch.long), \
               torch.tensor(input_lens, dtype=torch.long)
    y_tensor = torch.tensor(start_ids, dtype=torch.long), \
               torch.tensor(end_ids, dtype=torch.long)
    return x_tensor, y_tensor

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents

class Example(object):

    def __init__(self, article, abstract_sentences, vocab):
        # Get ids of special tokens
        start_decoding = vocab.word2id(START_DECODING)
        stop_decoding = vocab.word2id(STOP_DECODING)

        # Process the article
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        self.enc_input = [vocab.word2id(w) for w in
                          article_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        abstract = ' '.join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings
        abs_ids = [vocab.word2id(w) for w in
                   abstract_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding,
                                                        stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

# def get_input_from_batch(batch, use_cuda):
#     batch_size = len(batch.enc_lens)
#
#     enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
#     enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
#     enc_lens = batch.enc_lens
#     extra_zeros = None
#     enc_batch_extend_vocab = None
#
#     if config.pointer_gen:
#         enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
#         # max_art_oovs is the max over all the article oov list in the batch
#         if batch.max_art_oovs > 0:
#             extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
#
#     c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))
#
#     coverage = None
#     if config.is_coverage:
#         coverage = Variable(torch.zeros(enc_batch.size()))
#
#     if use_cuda:
#         enc_batch = enc_batch.cuda()
#         enc_padding_mask = enc_padding_mask.cuda()
#
#         if enc_batch_extend_vocab is not None:
#             enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
#         if extra_zeros is not None:
#             extra_zeros = extra_zeros.cuda()
#         c_t_1 = c_t_1.cuda()
#
#         if coverage is not None:
#             coverage = coverage.cuda()
#
#     return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage
#
#
# def get_output_from_batch(batch, use_cuda):
#     dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
#     dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
#     dec_lens = batch.dec_lens
#     max_dec_len = np.max(dec_lens)
#     dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()
#
#     target_batch = Variable(torch.from_numpy(batch.target_batch)).long()
#
#     if use_cuda:
#         dec_batch = dec_batch.cuda()
#         dec_padding_mask = dec_padding_mask.cuda()
#         dec_lens_var = dec_lens_var.cuda()
#         target_batch = target_batch.cuda()
#
#     return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch
