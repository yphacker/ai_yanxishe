# %%

import lightgbm as lgb

# %%

import pandas as pd
from tqdm.autonotebook import *
from bs4 import BeautifulSoup
import re

tqdm.pandas()

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

data = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
data = data.fillna(-1)


def salary_range_min(row):
    try:
        result = int(str(row['salary_range']).split('-')[0])
    except Exception:
        result = -1
    return result


def salary_range_max(row):
    try:
        result = int(str(row['salary_range']).split('-')[1])
    except Exception:
        result = -1
    return result


def location_2(row):
    try:
        result = str(row).split(',')[1]
    except Exception:
        result = '未知'
    return result


normal_feature = pd.DataFrame()
normal_feature['salary_min'] = data.progress_apply(lambda row: salary_range_min(row), axis=1)
normal_feature['salary_max'] = data.progress_apply(lambda row: salary_range_max(row), axis=1)
normal_feature['salary_median'] = (normal_feature['salary_max'] + normal_feature['salary_min']) / 2
normal_feature['salary_range'] = normal_feature['salary_max'] - normal_feature['salary_min']
normal_feature['telecommuting'] = list(data['telecommuting'])
normal_feature['has_company_logo'] = list(data['has_company_logo'])
normal_feature['has_questions'] = list(data['has_questions'])
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
normal_feature['employment_type'] = labelencoder.fit_transform(data['employment_type'].astype(str))
normal_feature['required_experience'] = labelencoder.fit_transform(data['required_experience'].astype(str))
normal_feature['required_education'] = labelencoder.fit_transform(data['required_education'].astype(str))
normal_feature['industry'] = labelencoder.fit_transform(data['industry'].astype(str))
normal_feature['function'] = labelencoder.fit_transform(data['function'].astype(str))

data['review'] = data.progress_apply(
    lambda row: str(row['title']) + ' ' + str(row['location']) + ' ' + str(row['company_profile']) + ' ' +
                str(row['description']) + ' ' + str(row['department']) + ' ' + str(row['requirements']) + ' ' + str(
        row['benefits']), axis=1)

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from tqdm import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df_train = data[:len(train)]
df_test = data[len(train):]

df_train['label'] = df_train['fraudulent'].astype(int)
data = pd.concat([df_train, df_test], axis=0, sort=False)
data['review'] = data['review'].apply(lambda row: str(row))

############################ tf-idf ############################
print('开始计算tf-idf特征')
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
discuss_tf = tf.fit_transform(data['review']).tocsr()
print('计算结束')

############################ 切分数据集 ##########################
print('开始进行一些前期处理')
train_feature = discuss_tf[:len(df_train)]
score = df_train['label']
test_feature = discuss_tf[len(df_train):]
print('处理完毕')


######################### 模型函数(返回sklean_stacking结果) ########################
def get_sklearn_classfiy_stacking(clf, train_feature, test_feature, score, model_name, class_number, n_folds, train_num,
                                  test_num):
    print('\n****开始跑', model_name, '****')
    stack_train = np.zeros((train_num, class_number))
    stack_test = np.zeros((test_num, class_number))
    score_mean = []
    skf = StratifiedKFold(n_splits=n_folds, random_state=1017)
    tqdm.desc = model_name
    for i, (tr, va) in enumerate(skf.split(train_feature, score)):
        clf.fit(train_feature[tr], score[tr])
        score_va = clf._predict_proba_lr(train_feature[va])
        score_te = clf._predict_proba_lr(test_feature)
        score_single = accuracy_score(score[va], clf.predict(train_feature[va]))
        score_mean.append(np.around(score_single, 5))
        stack_train[va] += score_va
        stack_test += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()
    df_stack['tfidf_' + model_name + '_classfiy_{}'.format(1)] = stack[:, 1]
    print(model_name, '处理完毕')
    return df_stack, score_mean


model_list = [
    ['LogisticRegression', LogisticRegression(random_state=1017, C=3)],
    ['SGDClassifier', SGDClassifier(random_state=1017, loss='log')],
    ['PassiveAggressiveClassifier', PassiveAggressiveClassifier(random_state=1017, C=2)],
    ['RidgeClassfiy', RidgeClassifier(random_state=1017)],
    ['LinearSVC', LinearSVC(random_state=1017)]
]

stack_feature = pd.DataFrame()
for i in model_list:
    stack_result, score_mean = get_sklearn_classfiy_stacking(i[1], train_feature, test_feature, score, i[0], 2, 8,
                                                             len(df_train), len(df_test))
    stack_feature = pd.concat([stack_feature, stack_result], axis=1, sort=False)
    print('五折结果', score_mean)
    print('平均结果', np.mean(score_mean))
normal_feature = pd.concat([stack_feature, normal_feature], axis=1, sort=False)

import pandas as pd

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# f1 = pd.read_csv('feature/normal_feature.csv')
# f2 = pd.read_csv('feature/w2v_feature.csv')
# f3 = pd.read_csv('feature/w2v_extend_feature.csv')

df_feature = normal_feature

train_feature = df_feature[:len(train)]
test_feature = df_feature[len(train):]

label = train['fraudulent'].astype(int)

import pandas as pd
from sklearn import model_selection
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, label, test_size=0.2,
                                                                    random_state=1017)
# train_feature = X_train
# label = Y_train

print('特征处理完毕......')

###################### lgb ##########################
import lightgbm as lgb

print('载入数据......')
lgb_train = lgb.Dataset(train_feature, label)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

print('开始训练......')
params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'verbose': 0,
    #             'metrics':{'binary_error'},
    #             'num_leaves':32,
    'objective': 'binary',
    #             'feature_fraction': 0.2,
    #             'bagging_fraction':0.7 ,
    'seed': 1024,
    'nthread': 50,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                verbose_eval=20,
                )

temp = gbm.predict(X_test)

print('结果：' + str(1 / (1 + mean_squared_error(Y_test, temp))))
print('特征重要性：' + str(list(gbm.feature_importance())))

y_test = gbm.predict(test_feature)
test_change_label = y_test.copy()

y_test_pos = np.argsort(y_test)

test_change_label[y_test_pos[:100]] = 0
test_change_label[y_test_pos[100:]] = 1
result = pd.DataFrame()
result['id'] = np.arange(0, len(y_test), 1)
result['result'] = np.around(test_change_label)
result['result'] = result['result'].astype(int)
result.to_csv(f'result/lgb.csv', index=False, header=None)
result.result.value_counts()

test_1 = test_change_label.copy()

# %%

import pandas as pd

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

f1 = pd.read_csv('feature/normal_feature.csv')
f2 = pd.read_csv('feature/w2v_feature.csv')

df_feature = pd.concat([f1, f2], axis=1, sort=False)

train_feature = df_feature[:len(train)]
test_feature = df_feature[len(train):]

label = train['fraudulent'].astype(int)

import pandas as pd
from sklearn import model_selection
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, label, test_size=0.2,
                                                                    random_state=1017)
# train_feature = X_train
# label = Y_train

print('特征处理完毕......')

###################### lgb ##########################
import lightgbm as lgb

print('载入数据......')
lgb_train = lgb.Dataset(train_feature, label)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

print('开始训练......')
params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'verbose': 0,
    #             'metrics':{'binary_error'},
    #             'num_leaves':32,
    'objective': 'binary',
    #             'feature_fraction': 0.2,
    #             'bagging_fraction':0.7 ,
    'seed': 1024,
    'nthread': 50,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=110,
                valid_sets=lgb_eval,
                verbose_eval=20,
                )

temp = gbm.predict(X_test)

print('结果：' + str(1 / (1 + mean_squared_error(Y_test, temp))))
print('特征重要性：' + str(list(gbm.feature_importance())))

y_test = gbm.predict(test_feature)
test_change_label = y_test.copy()

y_test_pos = np.argsort(y_test)

test_change_label[y_test_pos[:100]] = 0
test_change_label[y_test_pos[100:]] = 1
result = pd.DataFrame()
result['id'] = np.arange(0, len(y_test), 1)
result['result'] = np.around(test_change_label)
result['result'] = result['result'].astype(int)
result.to_csv(f'result/lgb.csv', index=False, header=None)

test_2 = test_change_label.copy()

# %%

import pandas as pd

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

f1 = pd.read_csv('feature/normal_feature.csv')
f2 = pd.read_csv('feature/w2v_feature.csv')
f3 = pd.read_csv('feature/w2v_extend_feature.csv')

df_feature = pd.concat([f1, f2, f3], axis=1, sort=False)

import random

random.seed(1024)
a = random.sample(list(df_feature.columns), 40)
df_feature = df_feature[a]

train_feature = df_feature[:len(train)]
test_feature = df_feature[len(train):]

label = train['fraudulent'].astype(int)

import pandas as pd
from sklearn import model_selection
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, label, test_size=0.2,
                                                                    random_state=1017)
# train_feature = X_train
# label = Y_train

print('特征处理完毕......')

###################### lgb ##########################
import lightgbm as lgb

print('载入数据......')
lgb_train = lgb.Dataset(train_feature, label)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

print('开始训练......')
params = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'verbose': 0,
    #             'metrics':{'binary_error'},
    #             'num_leaves':32,
    'objective': 'binary',
    #             'feature_fraction': 0.2,
    #             'bagging_fraction':0.7 ,
    'seed': 1024,
    'nthread': 50,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_eval,
                verbose_eval=20,
                )

temp = gbm.predict(X_test)

print('结果：' + str(1 / (1 + mean_squared_error(Y_test, temp))))
print('特征重要性：' + str(list(gbm.feature_importance())))

y_test = gbm.predict(test_feature)
test_change_label = y_test.copy()

y_test_pos = np.argsort(y_test)

test_change_label[y_test_pos[:100]] = 0
test_change_label[y_test_pos[100:]] = 1
result = pd.DataFrame()
result['id'] = np.arange(0, len(y_test), 1)
result['result'] = np.around(test_change_label)
result['result'] = result['result'].astype(int)
result.to_csv(f'result/lgb.csv', index=False, header=None)
test_3 = test_change_label.copy()

# %%

test_all = (test_1 + test_2 + test_3) / 3
result['id'] = np.arange(0, len(test_all), 1)
result['result'] = np.around(test_all)
result['result'] = result['result'].astype(int)
result.to_csv(f'result/vote.csv', index=False, header=None)

# %%
# nohup python lgb_model.py > info.txt 2>&1 &