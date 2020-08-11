# coding=utf-8
# author=yphacker

import jieba
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')


def get_text(row):
    if pd.isnull(row['question_title']):
        return row['question_detail']
    if pd.isnull(row['question_detail']):
        return row['question_title']
    return row['question_title'] + row['question_title']


train_df['text'] = train_df.apply(lambda x: get_text(x), axis=1)
test_df['text'] = test_df.apply(lambda x: get_text(x), axis=1)

train_df['text_jiaba'] = train_df.apply(lambda x: ' '.join(list(jieba.cut(x['text']))), axis=1)
test_df['text_jiaba'] = test_df.apply(lambda x: ' '.join(list(jieba.cut(x['text']))), axis=1)

word_vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    stop_words=["是", "的"],
    min_df=0.1,
    max_df=0.9,
    ngram_range=(1, 1),
    max_features=50)
# word_vectorizer.fit(data)
# word_features = word_vectorizer.transform(data)

# 数据向量化
print("Creating the tfidf vector...\n")
word_vectorizer.fit(train_df['text_jiaba'])
x_train = word_vectorizer.transform(train_df['text_jiaba'])
x_train = x_train.toarray()

x_test = word_vectorizer.transform(test_df['text_jiaba'])
x_test = x_test.toarray()

label_df = pd.read_csv('../data/topic2id.csv')
labels = label_df.topic_id.tolist()

from collections import defaultdict

labels_dict = defaultdict(set)
for i, row in train_df.iterrows():
    tag_ids = row['tag_ids'].split('|')
    for tag_id in tag_ids:
        labels_dict[tag_id].add(i)

outfile = open('../data/item_label.txt', 'w')
from tqdm import tqdm

for label, question_ids in tqdm(labels_dict.items()):
    y_train = np.zeros(train_df.shape[0])
    if not question_ids:
        #         outfile.write('{}\n'.format(','.join(['0']*train_df.shape[0])))
        continue
    for question_id in question_ids:
        y_train[question_id] = 1
    classifier = LogisticRegression(solver='liblinear')
    #     cv_score = np.mean(cross_val_score(classifier, x_train, y_train, cv=5, scoring='roc_auc'))
    #     scores.append(cv_score)
    #     print('cv score for class {} is {}'.format(label, cv_score))
    classifier.fit(x_train, y_train)
    preds = classifier.predict_proba(x_test)[:, 1]
    valid_label = dict()
    for question_id, pred in enumerate(preds):
        if pred >= 0.5:
            valid_label[question_id] = pred
    outfile.write('{}\t{}\n'.format(label, ';'.join([str(question_id) + ',' + str(pred)
                                                     for question_id, pred in valid_label.items()])))
outfile.close()
