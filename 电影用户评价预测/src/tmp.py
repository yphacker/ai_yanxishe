# coding=utf-8
# author=yphacker

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('../data/train.csv', sep='\t', names=header)

# 去重之后得到一个元祖，分别表示行与列,大小分别为943与1682
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

# 将样本分为训练集和测试集
train_data, test_data = train_test_split(df, test_size=0.25)

# 创建两个评分矩阵，即训练集与测试集矩阵
train_data_matrix = np.zeros((n_users, n_items))
# for循环，即在新生成的二维零矩阵中，将评分值赋值到二维矩阵中，因矩阵从0开始，所以[line[1]-1,line[2]-2]为新矩阵的横纵坐标
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

# 计算余弦相似性,使用sklearn的pairwise_distances函数来计算余弦相似性,同时创建了相似性矩阵
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


# 下列函数做出预测；评分矩阵，相似度，类型对象
def predict(ratings, similarity, type='user'):
    if type == 'user':
        # 求出用户打分的均值,axis=0表示纵向列取平均值,axis=1表示横向行上求和取均值
        mean_user_rating = ratings.mean(axis=1)
        # 可以百度np.newaxis用法，在此不再赘述
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# 计算得出相似度预测结果
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# 利用均方根误差进行评估
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
