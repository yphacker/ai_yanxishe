# 一些常规特征
import pandas as pd
from tqdm.autonotebook import *
from bs4 import BeautifulSoup
import re

tqdm.pandas()

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
data = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

# 将所有列进行拼接
data['review'] = data.progress_apply(
    lambda row: str(row['title']) + ' ' + str(row['location']) + ' ' + str(row['company_profile']) + ' ' +
                str(row['description']) + ' ' + str(row['department']) + ' ' + str(row['requirements']) + ' ' + str(
        row['benefits']), axis=1)

# %%

from gensim.models import Word2Vec
from collections import defaultdict

text_list = list(data['review'])

documents = text_list
texts = [[word for word in str(document).split(' ')] for document in documents]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] >= 5] for text in texts]

w2v = Word2Vec(texts, size=60, seed=1017)
w2v.wv.save_word2vec_format('model/w2v_128.txt')
print("w2v model done")

# %%


import gensim
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def get_w2v_avg(word2vec_Path, documents=documents):
    texts = documents
    w2v_dim = 60
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_Path, binary=False)
    vacab = model.vocab.keys()
    w2v_feature = np.zeros((len(texts), w2v_dim))
    w2v_feature_avg = np.zeros((len(texts), w2v_dim))

    for i, line in enumerate(texts):
        num = 0
        if line == 'null':
            w2v_feature_avg[i, :] = np.zeros(w2v_dim)
        else:
            for word in line.split():
                num += 1
                if word == "":
                    vec = np.zeros(w2v_dim)
                else:
                    vec = model[word] if word in vacab else np.zeros(w2v_dim)
                w2v_feature[i, :] += vec
            w2v_feature_avg[i, :] = w2v_feature[i, :] / num
    w2v_avg = pd.DataFrame(w2v_feature_avg)
    w2v_avg.columns = ['w2v_avg_' + str(i) for i in w2v_avg.columns]
    return w2v_avg


w2v_avg_feat = get_w2v_avg('model/w2v_128.txt')
print("done")


w2v_avg_feat.to_csv('feature/w2v_feature.csv', index=False)

# nohup python word2vec_feature.py > info.txt 2>&1 &

