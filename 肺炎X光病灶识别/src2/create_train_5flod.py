import pandas as pd

train= pd.read_csv("../data/train.csv",sep=",",header=None,names=['filename','label'])
print(train['label'].value_counts())
train['label'] = train['label'].apply(lambda x:x if x<=2 else 2)
# train['FileName'] = train['FileName'].apply(lambda x:str(x)+".jpg")
print(train.head(10))
print(train['label'].value_counts())
# train = pd.concat([train1,train2])
from sklearn.model_selection import StratifiedKFold

n_splits = 5
x = train['filename'].values
y = train['label'].values
skf = StratifiedKFold(n_splits=n_splits,random_state=250,shuffle=True)

for index,(train_index,test_index) in enumerate(skf.split(x,y)):
    res_train = pd.DataFrame()
    res_train['filename'] = train['filename'].iloc[train_index]
    res_train['label'] = train['label'].iloc[train_index]
    res_train.to_csv("data/train_{}.csv".format(index+1),index=False)

    res_train = pd.DataFrame()
    res_train['filename'] = train['filename'].iloc[test_index]
    res_train['label'] = train['label'].iloc[test_index]
    res_train.to_csv("data/test_{}.csv".format(index+1),index=False)