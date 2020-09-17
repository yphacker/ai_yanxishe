# -*- coding: utf-8 -*-

import pandas as pd
import time, datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split

itemInfo = pd.read_csv('../data/itemInfo.csv')
userInfo = pd.read_csv('../data/userInfo.csv')
train = pd.read_csv('../data/train.csv', header=None)
test = pd.read_csv('../data/test.csv', header=None)
train.columns = ['userId', 'movie_id', 'time', 'score']
test.columns = ['userId', 'movie_id', 'time']


def change(time1):
    s = []
    for i in time1:
        timeArray = time.localtime(i)
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        datetime1 = datetime.datetime.strptime(otherStyleTime, "%Y-%m-%d %H:%M:%S")
        s.append(datetime1)
    return s


itemInfo = itemInfo.drop(['IMDb_URL', 'movie_title'], axis=1)
itemInfo["release_date"] = pd.to_datetime(itemInfo["release_date"])
itemInfo["year"] = itemInfo["release_date"].dt.year
itemInfo["month"] = itemInfo["release_date"].dt.month
itemInfo["day"] = itemInfo["release_date"].dt.day
# itemInfo = itemInfo.drop(['release_date'],axis = 1)


# user
userInfo['useGender'] = userInfo['useGender'].map(lambda x: 1 if x == 'M' else 0)
s = list(userInfo['useOccupation'])
sets = set(s)
user_dict = {'administrator': 0,
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
ss_code = []
for i in range(len(s)):
    ss_code.append(user_dict[s[i]])
userInfo['useOccupation'] = ss_code
userInfo = userInfo.drop(['useZipcode'], axis=1)

train.iloc[:, 2] = change(list(train.iloc[:, 2]))
test.iloc[:, 2] = change(list(test.iloc[:, 2]))
Y = train['score']
train = train.drop(['score'], axis=1)

train['time_year'] = train["time"].dt.year
train['time_month'] = train["time"].dt.month
train['time_day'] = train["time"].dt.day
test['time_year'] = test["time"].dt.year
test['time_month'] = test["time"].dt.month
test['time_day'] = test["time"].dt.day

# concat
big_train = pd.merge(train, userInfo, on='userId', how='left')
big_train = pd.merge(big_train, itemInfo, on='movie_id', how='left')

big_test = pd.merge(test, userInfo, on='userId', how='left')
big_test = pd.merge(big_test, itemInfo, on='movie_id', how='left')

big_train['between'] = (big_train['time'] - big_train['release_date']).dt.days
big_test['between'] = (big_test['time'] - big_test['release_date']).dt.days
#
big_train = big_train.drop(['time', 'release_date'], axis=1)
big_test = big_test.drop(['time', 'release_date'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(big_train, Y, test_size=0.1, random_state=42)
trn_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_test, y_test, reference=trn_data)
### 设置初始参数--不含交叉验证参数
print('设置参数')
params = {'num_leaves': 100,
          'max_depth': 50,
          'max_bin': 255,
          'min_data_in_leaf': 11,
          'feature_fraction': 0.7,
          'bagging_fraction': 0.6,
          'bagging_freq': 5,
          'lambda_l1': 1e-5,
          'lambda_l2': 0,
          'min_split_gain': 0.1,
          'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'metric': 'multi_logloss',
          'nthread': 5,
          'learning_rate': 0.1,
          'num_class': 6,
          "verbosity": -1
          }

clf = lgb.train(params,
                trn_data,
                num_boost_round=2200,
                verbose_eval=20)

Y_pred = clf.predict(big_test)
Y_pred = Y_pred.argmax(axis=1)
test = pd.read_csv('../data/test.csv', header=None)
test.columns = ['userId', 'movie_id', 'time']
sub = test[['userId', 'movie_id', 'time']]
sub['label'] = Y_pred
# 输出结果
sub.to_csv('subs.csv', index=None, header=None)
