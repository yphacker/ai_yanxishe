# coding=utf-8
# author=yphacker

import numpy as np
from conf import config
from conf import bert_model_config
from utils.bert import tokenization


def get_bert_param_lists(texts):
    """
    将数据转换成Bert能够使用的格式
    input_ids：根据BERT-Base-Chinese checkpoint中的vocabtxt中每个字出现的index，将训练文本中的每一个字替换为vocab.txt中的index，需要添加开始CLS和结束SEP
    input_masks：包含开始CLS和结束SEP有字就填1
    segment_ids：seq2seq类任务同时传入两句训练关联训练数据时，有意义，传入一句训练数据则都为0
    以上三个list需要用0补齐到max_seq_length的长度
    """
    # token 处理器，主要作用就是 分字，将字转换成ID。vocab_file 字典文件路径
    tokenizer = tokenization.FullTokenizer(vocab_file=bert_model_config.bert_vocab_path)
    input_ids_list = []
    input_masks_list = []
    segment_ids_list = []
    for text_a, text_b in texts:
        single_input_id, single_input_mask, single_segment_id = \
            convert_single_example_simple(config.max_seq_length, tokenizer, text)
        input_ids_list.append(single_input_id)
        input_masks_list.append(single_input_mask)
        segment_ids_list.append(single_segment_id)
    input_ids = np.asarray(input_ids_list, dtype=np.int32)
    input_masks = np.asarray(input_masks_list, dtype=np.int32)
    segment_ids = np.asarray(segment_ids_list, dtype=np.int32)
    return input_ids, input_masks, segment_ids


def bert_bacth_iter(x, y, batch_size=config.batch_size):
    input_ids, input_masks, segment_ids = x
    index = np.random.permutation(len(y))
    n_batches = len(y) // batch_size
    for batch_index in np.array_split(index, n_batches):
        batch_input_ids, batch_input_masks, batch_segment_ids, batch_y = \
            input_ids[batch_index], input_masks[batch_index], segment_ids[batch_index], y[batch_index]
        yield (batch_input_ids, batch_input_masks, batch_segment_ids), batch_y


def batch_iter(x, y, batch_size=config.batch_size):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def get_sequence_length(x_batch):
    """
    Args:
        x_batch:a batch of input_data
    Returns:
        sequence_lenghts: a list of acutal length of  every senuence_data in input_data
    """
    sequence_lengths = []
    for x in x_batch:
        actual_length = np.sum(np.sign(x))
        sequence_lengths.append(actual_length)
    return sequence_lengths


def export_word2vec_vectors(vocab):
    """
    Args:
        vocab: word_to_id
        word2vec_dir:file path of have trained word vector by word2vec
        trimmed_filename:file path of changing word_vector to numpy file
    Returns:
        save vocab_vector to numpy file

    """
    infile = open(config.vector_word_filename, 'r')
    voc_size, vec_dim = None, None
    embeddings = None
    for i, line in enumerate(infile):
        if i == 0:
            voc_size, vec_dim = map(int, line.split(' '))
            embeddings = np.zeros([len(vocab), vec_dim])
            continue
        items = line.split(' ')
        word = items[0]
        # vec = np.asarray(items[1:], dtype='float32')
        vec = np.asarray(items[1:])
        if word in vocab:
            word_idx = vocab[word]
            embeddings[word_idx] = np.asarray(vec)
    np.savez_compressed(config.vector_word_npz, embeddings=embeddings)
    infile.close()


def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]


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


def convert_single_example_simple(max_seq_length, tokenizer, text_a, text_b=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)  # 这里主要是将中文分字
    if tokens_b:
        # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
        # 因为要为句子补上[CLS], [SEP], [SEP]
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 2
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
    # (a) 两个句子:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) 单个句子:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # 这里 "type_ids" 主要用于区分第一个第二个句子。
    # 第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
    # 因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将中文转换成ids
    # 创建mask
    input_mask = [1] * len(input_ids)
    # 对于输入进行补0
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids  # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数
