import pandas as pd
import json

with open("/home/detao/Videos/mmdetection_master/demo/result_20191211144817.json","r") as f:
    load_dict = json.load(f)


res= pd.DataFrame()
dict = {}
for i in range(6671):
    dict[i] = 0
jl = 0
for i in load_dict:

    if (i['score']>=0.4)and(i["category"]==1):
        jl =jl +1
        print(i)
        dict[int(i['name'].split('.')[0])] = dict[int(i['name'].split('.')[0])] + 1
print(jl)
filename = []
label = []
for (k,v) in dict.items():
    filename.append(k)
    label.append(v)
res['filename'] = filename
res['label'] = label
print(res['label'].value_counts())
res.to_csv("res.csv",index=False,header=None)
print(jl)


