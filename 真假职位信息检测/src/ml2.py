# coding=utf-8
# author=yphacker

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

train_df = pd.read_csv("../data/train.csv")
train_df = train_df.fillna("NAN None")
train_df['text'] = train_df['description'] + train_df['requirements'] + train_df['benefits']

test_df = pd.read_csv("../data/test.csv")
test_df = test_df.fillna("NAN None")
test_df['text'] = test_df['description'] + test_df['requirements'] + test_df['benefits']

columns = train_df.columns.tolist()
columns.remove('fraudulent')
print(columns)
x_train = train_df[columns]
y_train = train_df['fraudulent']
x_test = test_df
print(x_train.shape)
print(x_test.shape)

model = CatBoostClassifier(task_type='GPU', scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train))
model.fit(x_train, y_train, cat_features=[8, 9, 10], text_features=[0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16])

probs = model.predict(x_test)
submisson = pd.DataFrame({'index': range(len(probs)), 'label': [1 if x > 0.5 else 0 for x in probs]})
submisson.to_csv("submission.csv", index=False, header=False)
