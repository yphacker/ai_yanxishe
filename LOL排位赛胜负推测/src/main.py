# coding=utf-8
# author=yphacker


from xgboost.sklearn import XGBClassifier
from fastai.tabular import *

train_df = pd.read_csv('../data/train_df.csv')
test_df = pd.read_csv('../data/test_df.csv')
train_x = train_df[train_df.columns[1:]]

train_y = train_df.label
test_x = test_df

model = XGBClassifier(
    num_leaves=64,
    max_depth=7,
    learning_rate=0.1,
    n_estimators=401,
    subsample=0.6,
    feature_fraction=0.6,
    reg_alpha=8,
    reg_lambda=12,
    random_state=1983,
    # tree_method='gpu_hist',
)
# gs = GridSearchCV(model,  scoring='accuracy', cv=5, verbose=1, n_jobs=-1 )
# results = gs.fit(train_x, train_y)
# print("BEST PARAMETERS: " + str(results.best_params_))
# print("BEST CV SCORE: " + str(results.best_score_))
model.fit(train_x, train_y)

pre = model.predict(test_x)
sub_df = pd.DataFrame()
sub_df['id'] = [146022 + i for i in range(len(test_df))]
sub_df['label'] = pre
sub_df.to_csv('submission.csv', index=False, header=False)
