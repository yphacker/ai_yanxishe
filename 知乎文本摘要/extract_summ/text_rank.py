# coding=utf-8
# author=yphacker


import jieba
import math
from heapq import nlargest
from itertools import product, count
import numpy as np


def cut_text(text):
    import re
    from utils.utils import load_stopwords
    stopwords = load_stopwords()
    sentence_list = []
    # 把元素按照[。！；？]进行分隔，得到句子。
    line_split = re.split(r'[。！；？]', text)
    # [。！；？]这些符号也会划分出来，把它们去掉。
    sentence_list = [line.strip() for line in line_split if
                     line.strip() not in ['。', '！', '？', '；'] and len(line.strip()) > 1]

    def cut_sentence(sentence):
        sentence = re.sub(r'[^\u4e00-\u9fa5]+', '', sentence)
        sentence_cut = jieba.cut(sentence)
        word_list = []
        for word in sentence_cut:
            if word not in stopwords:
                word_list.append(word)
                # 如果句子整个被过滤掉了，如：'02-2717:56'被过滤，那就返回[],保持句子的数量不变
        return word_list

    sentence_word_list = []
    for sentence in sentence_list:
        tmp = cut_sentence(sentence)
        sentence_word_list.append(tmp)
    if len(sentence_list) != len(sentence_word_list):
        Exception('{}, 分词之后有问题'.format(text))
    return sentence_list, sentence_word_list


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
    for i, j in product(range(num), repeat=2):  # range(num)未0-15的整数
        if i != j:
            board[i][j] = compute_similarity_by_avg(model, word_sent[i], word_sent[j])
    return board


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


# 冒泡排序
def bubble_sort(array):
    for i in range(len(array)):
        for j in range(i, len(array)):
            if array[i] > array[j]:
                array[i], array[j] = array[j], array[i]
    return array


# 生成摘要
def summarize(model, text, n, stoppath):
    tokens = cut_sentences(text)  # 按句切分，这是一个generator
    sentences = []
    sents = []
    for sent in tokens:  # 每句话
        sentences.append(sent)  # 把每句话一次加入sentences,也即把原来的generator变成数组，每个元素是一个句子。
        # 把分词后的每个词加入sents，也即把原来的generator变成j矩阵，每行是一句，单个字成列。
        sents.append([word for word in jieba.cut(sent) if word])
    sents = filter_symbols(sents, stoppath)
    sents = filter_model(model, sents)  # 去除模型外的词
    graph = create_graph(model, sents)  # 传入句子链表  返回句子之间相似度的图
    scores = weight_sentences_rank(graph)  # 输入相似度的图（矩阵),返回各个句子的分数
    sent_selected = nlargest(n, zip(scores, count()))  # 选择出得分最大的n个句子的值和编号
    sent_index = []
    if len(sent_selected) < n:
        return None
    for i in range(n):
        sent_index.append(sent_selected[i][1])
    sent_index = bubble_sort(sent_index)  # 按文中顺序排序
    summarystr = ""
    for sum in summary:
        summarystr = summarystr + sum
    # return summarystr
    return summarystr
    # return [sentences[i] for i in sent_index]


# def Model(modelpath):
#     model = word2vec.Word2Vec.load(modelpath)  # 加载训练好的模型
#     return model

# 利用第三方包
def summarize2(sentences_list, sim_mat, summarizenum):
    if len(sim_mat) <= summarizenum:
        return ''.join(sentences_list)
    import networkx as nx

    # 利用句子相似度矩阵构建图结构，句子为节点，句子相似度为转移概率
    nx_graph = nx.from_numpy_array(sim_mat)

    # 得到所有句子的textrank值
    scores = nx.pagerank(nx_graph)

    # 根据textrank值对未处理的句子进行排序
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences_list)), reverse=True)
    summarize = ''
    for i in range(summarizenum):
        summarize += ranked_sentences[i][1]
    return summarize


def main(summarizenum=2):
    import pandas as pd
    from conf import config
    from utils.utils import load_word_embeddings, get_sim_mat
    from sklearn.metrics.pairwise import cosine_similarity
    # train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    texts = test_df['article']
    pred_list = []
    word_embeddings = load_word_embeddings()
    for text in texts:
        sentence_list, sentence_words_list = cut_text(text)
        sentence_vectors = []
        # 词向量的平均，作为该句子的向量
        for sentence_words in sentence_words_list:
            if len(sentence_words) != 0:
                # 如果句子中的词语不在字典中，那就把embedding设为300维元素为0的向量。
                # 得到句子中全部词的词向量后，求平均值，得到句子的向量表示
                v = sum([word_embeddings.get(w, np.zeros((300,))) for w in sentence_words]) / (len(sentence_words))
            else:
                # 如果句子为[]，那么就向量表示为300维元素为0个向量。
                v = np.zeros((300,))
            sentence_vectors.append(v)
        sim_mat = get_sim_mat(sentence_list, sentence_vectors)
        print("句子相似度矩阵的形状为：", sim_mat.shape)
        pred = summarize2(sentence_list, sim_mat, summarizenum)
        pred_list.append(pred)
    submission_df = pd.DataFrame({'id': range(len(pred_list)), 'summarization': pred_list})
    submission_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
