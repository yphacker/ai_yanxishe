# coding=utf-8
# author=yphacker

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

job_dict = {'administrator': 0,
            'artist': 1,
            'doctor': 2,
            'educator': 3,
            'engineer': 4,
            'entertainment': 5,
            'executive': 6,
            'healthcare': 7,
            'homemaker': 8,
            'lawyer': 9,
            'librarian': 10,
            'marketing': 11,
            'none': 12,
            'other': 13,
            'programmer': 14,
            'retired': 15,
            'salesman': 16,
            'scientist': 17,
            'student': 18,
            'technician': 19,
            'writer': 20}
gender_dict = {'F': 0, 'M': 1}
# user_df = pd.read_table('../data/userInfo.csv', sep=',')
# print(user_df['useOccupation'].value_counts())
# user_df['job_id'] = user_df['useOccupation'].map(job_dict)
# user_df['gender_id'] = user_df['useGender'].map(gender_dict)
# user_df = user_df[['userId', 'useAge', 'gender_id', 'job_id']]
# print(user_df.head())

import re
item_df = pd.read_table('../data/itemInfo.csv', sep=',')
movies = item_df[['movie_id', 'movie_title', 'release_date']]
movies_orig = item_df.values

# 将Title中的年份去掉
pattern = re.compile(r'^(.*)\((\d+)\)')
print(movies.head())
# title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['movie_title']))}
# title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['movie_title']))}
for i, val in enumerate(set(movies['movie_title'])):
    tmp = pattern.match(val)
    if tmp:
        pass
    else:
        print(val)
# movies['movie_title'] = movies['movie_title'].map(title_map)

# # 电影类型转数字字典
# genres_set = set()
# for val in movies['release_date'].str.split('|'):
#     genres_set.update(val)
#
# genres_set.add('<PAD>')
# genres2int = {val: ii for ii, val in enumerate(genres_set)}
#
# # 将电影类型转成等长数字列表，长度是18
# genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}
#
# for key in genres_map:
#     for cnt in range(max(genres2int.values()) - len(genres_map[key])):
#         genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
#
# movies['release_date'] = movies['release_date'].map(genres_map)
#
# # 电影Title转数字字典
# title_set = set()
# for val in movies['movie_title'].str.split():
#     title_set.update(val)
#
# title_set.add('<PAD>')
# title2int = {val: ii for ii, val in enumerate(title_set)}
#
# # 将电影Title转成等长数字列表，长度是15
# title_count = 15
# title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['movie_title']))}
