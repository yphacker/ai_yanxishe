# coding=utf-8
# author=yphacker

import logging
from gensim.models import word2vec

wikipath = u"/data01/chennan/wiki_chs.txt"
modeloutpath = '/data01/chennan/modelcn'

'''可执行语句'''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(wikipath)  # 加载语料
model = word2vec.Word2Vec(sentences, size=200, min_count=2)     # 训练
model.save(modeloutpath)      # 模型存储
