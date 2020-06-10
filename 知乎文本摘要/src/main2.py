# coding=utf-8
# author=yphacker

from ceshi import get_text

import pandas as pd

df = pd.read_csv('../data/test.csv')
preds_list = []
for i, row in df.iterrows():
    if i >= 50:
        break
    preds_list.append(get_text(row['article']))
submission = pd.DataFrame({"id": range(len(preds_list)), "summarization": preds_list})
submission.to_csv('submission.csv', index=False, header=False)
