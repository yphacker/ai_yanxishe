# coding=utf-8
# author=yphacker

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from conf import config

train_df = pd.read_csv('{}/train.csv'.format(config.orig_data_path))
test_df = pd.read_csv('{}/test.csv'.format(config.orig_data_path))
role_df = pd.read_csv('{}/role.csv'.format(config.orig_data_path))
match_df = pd.read_csv('{}/matches.csv'.format(config.orig_data_path))
champ_df = pd.read_csv('{}/champs.csv'.format(config.orig_data_path))

role_df = role_df.fillna(0.0)
role_df.loc[role_df['wardsbought'] == '\\N', 'wardsbought'] = 99

role_df['kDa'] = (role_df['kills'] + role_df['assists']) / (role_df['deaths'] + 1e-3)
role_df['heal_dmg_ratio'] = role_df['totheal'] / (role_df['totdmgtaken'] + 1e-3)
role_df['gold_earn_spent_diff'] = role_df['goldearned'] - role_df['goldspent']
role_df['truedmgtochamp_ratio'] = role_df['truedmgtochamp'] / (role_df['totdmgtochamp'] + 1e-3)
role_df['dmgtoturrets_ratio'] = role_df['dmgtoturrets'] / (role_df['dmgtoobj'] + 1e-3)

# 每个player在比赛的总经济
role_df['economy'] = np.sum(role_df[['item1', 'item2', 'item3', 'item4', 'item5', 'item6']].values, axis=1)


# 对位置进行编码
def func(x):
    if x == 'BOT':
        return 0
    elif x == 'JUNGLE':
        return 1
    elif x == 'MID':
        return 2
    else:
        return 4


role_df['position_code'] = role_df.position.apply(func)


# 对玩家常用角色进行编码
def func(x):
    if x == 'SOLO':
        return 0
    elif x == 'NONE':
        return 1
    elif x == 'DUO_CARRY':
        return 2
    elif x == 'DUO_SUPPORT':
        return 4
    else:
        return 8


role_df['role_code'] = role_df.role.apply(func)

role_df.wardsbought = role_df.wardsbought.astype(np.float64)
role_df.info()

# 将训练集和测试集合并
all_df = pd.concat([train_df, test_df], axis=0)

# 构建统计特征
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# import numpy_gpu
def create_numeric_features(col_name):
    values = role_df[col_name].values.reshape(-1, 10)
    team1, team2 = values[:, :5], values[:, 5:]
    all_df[f'{col_name}_max_diff'] = np.max(team1, axis=1) - np.max(team2, axis=1)
    all_df[f'{col_name}_max_ratio'] = np.max(team1, axis=1) / (np.max(team2, axis=1) + 1e-5)

    all_df[f'{col_name}_median_diff'] = np.median(team1, axis=1) - np.median(team2, axis=1)
    all_df[f'{col_name}_median_ratio'] = np.median(team1, axis=1) / (np.median(team2, axis=1) + 1e-5)

    all_df[f'{col_name}_sum_diff'] = np.sum(team1, axis=1) - np.sum(team2, axis=1)
    all_df[f'{col_name}_sum_ratio'] = np.sum(team1, axis=1) / (np.sum(team2, axis=1) + 1e-5)

    all_df[f'{col_name}_min_diff'] = np.min(team1, axis=1) - np.min(team2, axis=1)
    all_df[f'{col_name}_min_ratio'] = np.min(team1, axis=1) / (np.min(team2, axis=1) + 1e-5)

    all_df[f'{col_name}_std_diff'] = np.std(team1, axis=1) - np.std(team2, axis=1)
    all_df[f'{col_name}_std_ratio'] = np.std(team1, axis=1) / (np.std(team2, axis=1) + 1e-5)

    #     all_df[f'{col_name}_l1_dis'] = mean_absolute_error(team1.T, team2.T, multioutput='raw_values')
    #     all_df[f'{col_name}_l2_dis'] = mean_squared_error(team1.T, team2.T, multioutput='raw_values')
    #     all_df[f'{col_name}_r2_score'] = r2_score(team1.T, team2.T, multioutput='raw_values')
    return all_df


for col_name in tqdm(role_df.columns.values[13:]):
    if col_name in ['totdmgdealt', 'truedmgdealt', 'timecc', 'wardsbought']:
        continue
    print(col_name)
    create_numeric_features(col_name)

# 两队是否阵容合理，前述对位置进行了编码 0：BOT, 1:JUNGLE, 2:MID, 4:TOP， 如果阵容合理应该和为7
# 红色方合理 1， 不合理 0， 蓝色方合理 4， 不合理 2
position = role_df.position_code.values.reshape(-1, 10)
blue_is_reasonable = np.sum(position[:, :5], axis=1) == 7
red_is_reasonable = np.sum(position[:, 5:], axis=1) == 7
print(red_is_reasonable.sum(), blue_is_reasonable.sum())
blue_is_reasonable = np.array([1 if item else 0 for item in blue_is_reasonable])
red_is_reasonable = np.array([4 if item else 2 for item in red_is_reasonable])
all_df['is_reasonable'] = blue_is_reasonable + red_is_reasonable
print(red_is_reasonable.sum(), blue_is_reasonable.sum())
all_df['is_reasonable'].value_counts()

# 比赛持续时间
all_df['duration'] = match_df['duration']
all_df.shape

train_x = all_df[:len(train_df)][all_df.columns.values[2:]]
train_y = all_df[:len(train_df)][all_df.columns.values[1]]
test_x = all_df[len(train_df):][all_df.columns.values[2:]]
del all_df

train_x.shape, train_y.shape, test_x.shape

_train_df = pd.concat([train_y, train_x], axis=1)
_train_df.to_csv('../data/train_df.csv', index=None)
test_x.to_csv('../data/test_df.csv', index=None)
