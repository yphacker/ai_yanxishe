import time
import copy
import pandas as pd
from collections import defaultdict
import numpy as np

def doit(str):
    for i in range(5-len(str)):
        str = '0'+str
    return str+'.jpg'

result_ratios = defaultdict(lambda: 0)
mode="test"
mt = 5#model_total
model_name = 'efficientnetb7_500_all'


result_ratios['{}_1'.format(model_name)] = 1.0/mt
result_ratios['{}_2'.format(model_name)] = 1.0/mt
result_ratios['{}_3'.format(model_name)] = 1.0/mt
result_ratios['{}_4'.format(model_name)] = 1.0/mt
result_ratios['{}_5'.format(model_name)] = 1.0/mt

print( sum(result_ratios.values()))

def str2np(str):
    return np.array([float(x) for x in str.split(";")])
def np2str(arr):
    return ";".join(["%.16f" % x for x in arr])
for index, model in enumerate(result_ratios.keys()):
    print('ratio: %.3f, model: %s' % (result_ratios[model], model))
    result = pd.read_csv('result/{}_{}/submission.csv'.format(model_name,index+1),sep=",")
    # result = result.sort_values(by='filename').reset_index(drop=True)
    result['probability'] = result['probability'].apply(lambda x: str2np(x))
    print(result.head(3))

    if index == 0:
        ensembled_result = copy.deepcopy(result)
        ensembled_result['probability'] =0

    ensembled_result['probability'] = ensembled_result['probability'] + result['probability']*result_ratios[model]/4

    result = pd.read_csv('result/{}_{}/submission_hflip.csv'.format(model_name, index + 1), sep=",")
    result['probability'] = result['probability'].apply(lambda x: str2np(x))
    ensembled_result['probability'] = ensembled_result['probability'] + result['probability'] * result_ratios[model] / 4

    result = pd.read_csv('result/{}_{}/submission_vflip.csv'.format(model_name, index + 1), sep=",")
    result['probability'] = result['probability'].apply(lambda x: str2np(x))
    ensembled_result['probability'] = ensembled_result['probability'] + result['probability'] * result_ratios[model] / 4

    result = pd.read_csv('result/{}_{}/submission_vhflip.csv'.format(model_name, index + 1), sep=",")
    result['probability'] = result['probability'].apply(lambda x: str2np(x))
    ensembled_result['probability'] = ensembled_result['probability'] + result['probability'] * result_ratios[model] / 4

    print(ensembled_result.head(3))

lis = {}
lis[1] = "airplane"
lis[2] = "ship"
lis[3] = "bridge"
lis[4] = "oilcan"
lis[5] = "build"


ensembled_result['type'] = ensembled_result['probability'].apply(lambda x:np.argmax(x))
ensembled_result['FileName'] = ensembled_result['FileName'].apply(lambda x:x[:-4])
# ensembled_result['probability'] = ensembled_result['probability'].apply(lambda x: np2str(x))
ensembled_result[['FileName','type']].to_csv("result/self_merge/{}_selfmerge.csv".format(model_name),index=False,header=None)
ensembled_result['probability'] = ensembled_result['probability'].apply(lambda x: np2str(x))
ensembled_result.to_csv("result/self_merge/{}_selfmerge_prob.csv".format(model_name),index=False)
print(ensembled_result.shape)
#
