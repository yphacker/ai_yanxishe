# coding=utf-8
# author=yphacker

'''
Interface: ExtractiveCN(text,  summarizenum=4, stoppath='stopword.txt')
SupportFile: stopword.txt  model
Function: 中文抽取式自动摘要
Algorithm: word2vec + Textrank
'''

import jieba
import math
from heapq import nlargest
from itertools import product, count
import numpy as np
from gensim.models import word2vec

np.seterr(all='warn')


# 按句切分并返回一个generator
# note：切分方法很重要，可以有效避免同一句话中间转折，从而导致被断句。这种方法先replace('\n','')，再检测。！？'''
def cut_sentences(sentence):
    puns = frozenset(u'。！？')   # 创建不可变集合{'。', '！', '？'}
    tmp = []
    for ch in sentence:   # 每个字符
        tmp.append(ch)   # 加入tmp末尾
        if puns.__contains__(ch):   # 如果某个字是。！？
            yield ''.join(tmp)      # 完成一次迭代也即一句话为一个元素
            tmp = []                # 重置tmp
    yield ''.join(tmp)     # 返回一个generator


# 创建停用词表
def create_stopwords(stoppath):
    stop_list = [line.strip() for line in open(stoppath, 'r', encoding='utf-8').readlines()]
    return stop_list


# 计算两个句子的相似性
def two_sentences_similarity(sents_1, sents_2):
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    return counter / (math.log(len(sents_1) + len(sents_2)))


# 传入句子链表  返回句子之间相似度的图
def create_graph(model, word_sent):
    num = len(word_sent)
    board = [[0.0 for _ in range(num)] for _ in range(num)]
    for i, j in product(range(num), repeat=2):      # range(num)未0-15的整数
        if i != j:
            board[i][j] = compute_similarity_by_avg(model, word_sent[i], word_sent[j])
    return board


# 计算两个向量之间的余弦相似度
def cosine_similarity(vec1, vec2):
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


# 对两个句子求平均词向量
def compute_similarity_by_avg(model, sents_1, sents_2):
    if len(sents_1) == 0 or len(sents_2) == 0:
        return 0.0
    vec1 = model[sents_1[0]]
    for word1 in sents_1[1:]:
        vec1 = vec1 + model[word1]
    vec2 = model[sents_2[0]]
    for word2 in sents_2[1:]:
        vec2 = vec2 + model[word2]
    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
    return similarity


# 计算句子在图中的分数
def calculate_score(weight_graph, scores, i):
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0
    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
            if denominator == 0:
                denominator = 1
        added_score += fraction / denominator
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score


# 输入相似度的图（矩阵),返回各个句子的分数
def weight_sentences_rank(weight_graph):
    # 初始分数设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores


# 判断前后分数有无变化
def different(scores, old_scores):

    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            flag = True
            break
    return flag


# 过滤符号
def filter_symbols(sents, stoppath):
    stopwords = create_stopwords(stoppath) + ['。', ' ', '.']
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word in stopwords:
                sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents


# 过滤掉模型外未训练到的字
def filter_model(model, sents):
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word not in model.wv.vocab:
                # print('remove' + word)
                sentence.remove(word)       # 剔除没有在训练模型中的字
        if sentence:
            _sents.append(sentence)     # 剔除后加入_sent并返回
    return _sents


# 冒泡排序
def bubble_sort(array):
    for i in range(len(array)):
        for j in range(i, len(array)):
            if array[i] > array[j]:
                array[i], array[j] = array[j], array[i]
    return array


# 生成摘要
def summarize(model, text, n, stoppath):
    tokens = cut_sentences(text)    # 按句切分，这是一个generator
    sentences = []
    sents = []
    for sent in tokens:     # 每句话
        sentences.append(sent)      # 把每句话一次加入sentences,也即把原来的generator变成数组，每个元素是一个句子。
        sents.append([word for word in jieba.cut(sent) if word])        # 把分词后的每个词加入sents，也即把原来的generator变成j矩阵，每行是一句，单个字成列。
    sents = filter_symbols(sents, stoppath)
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词
    sents = filter_model(model, sents)     # 去除模型外的词，一次并不成功
    graph = create_graph(model, sents)     # 传入句子链表  返回句子之间相似度的图
    scores = weight_sentences_rank(graph)       # 输入相似度的图（矩阵),返回各个句子的分数
    sent_selected = nlargest(n, zip(scores, count()))       # 选择出得分最大的n个句子的值和编号
    sent_index = []
    if len(sent_selected) < n:
        return None
    for i in range(n):
        sent_index.append(sent_selected[i][1])
    sent_index = bubble_sort(sent_index)        # 按文中顺序排序
    return [sentences[i] for i in sent_index]


def Model(modelpath):
    model = word2vec.Word2Vec.load(modelpath)  # 加载训练好的模型
    return model


def ExtractiveCN(text, model, summarizenum=4, stoppath='stopword.txt'):
    # model = Model(modelpath)
    summarize(model, text, summarizenum, stoppath)
    summary = summarize(model, text, summarizenum, stoppath)   # 摘要summarizenum句
    if summary is None:
        return text
    summarystr = ""
    for sum in summary:
        summarystr = summarystr + sum
    return summarystr

