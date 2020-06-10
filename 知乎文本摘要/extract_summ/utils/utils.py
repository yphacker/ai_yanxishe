# coding=utf-8
# author=yphacker

import numpy as np
from gensim.models import word2vec
from sklearn.metrics.pairwise import cosine_similarity
from conf import config


# sentences = []
# model = word2vec.Word2Vec(sentences, size=300, min_count=2)     # 训练
# model.save('../data/word2vec.bin')      # 模型存储

def load_word_embeddings():
    word_embeddings = {}
    infile = open('{}/sgns.zhihu.word'.format(config.data_path), 'r')
    for i, line in enumerate(infile):
        if i == 0:
            continue
        values = line.split()
        # 第一个元素是词语
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = embedding
    infile.close()
    print("一共有" + str(len(word_embeddings)) + "个词语/字。")
    return word_embeddings


# 创建停用词表
def load_stopwords(stoppath='../data/stopwords.txt'):
    stop_list = [line.strip() for line in open(stoppath, 'r', encoding='utf-8').readlines()]
    return stop_list


# 计算两个向量之间的余弦相似度
def cosine_sim(vec1, vec2):
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


# 对两个句子求平均词向量
def compute_similarity_by_avg(vec1, vec2):
    similarity = cosine_sim(vec1 / len(vec1), vec2 / len(vec2))
    return similarity


def get_sim_mat(sentence_list, sent_vectors):
    sim_mat = np.zeros([len(sentence_list), len(sentence_list)])
    for i in range(len(sentence_list)):
        for j in range(len(sentence_list)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sent_vectors[i].reshape(1, 300),
                                                  sent_vectors[j].reshape(1, 300))[0, 0]
    return sim_mat


if __name__ == '__main__':
    word_embeddings = load_word_embeddings()
