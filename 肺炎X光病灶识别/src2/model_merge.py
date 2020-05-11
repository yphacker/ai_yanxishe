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
mt =5#model_total
model_name = 'ensemble_1222'



result_ratios['resnet152_500_all'] = 0.1
result_ratios['seresnext101_500_all'] = 0.1
result_ratios['seresnext101_768_all'] = 0.1
result_ratios['seresnet152_500_all'] = 0.1
result_ratios['senet154_500_all'] = 0.3
result_ratios['efficientnetb7_500_all'] = 0.3
# result_ratios['resnet50_896_all'] = 1.0/mt
# result_ratios['efficientnet_768_all'] = 1.0/mt





print( sum(result_ratios.values()))

def str2np(str):
    return np.array([float(x) for x in str.split(";")])
def np2str(arr):
    return ";".join(["%.16f" % x for x in arr])
for index, model in enumerate(result_ratios.keys()):
    print('ratio: %.3f, model: %s' % (result_ratios[model], model))
    result = pd.read_csv('result/self_merge/{}_selfmerge_prob.csv'.format(model),sep=",")
    # result = result.sort_values(by='filename').reset_index(drop=True)
    result['probability'] = result['probability'].apply(lambda x: str2np(x))
    print(result.head(3))

    if index == 0:
        ensembled_result = copy.deepcopy(result)
        ensembled_result['probability'] =0

    ensembled_result['probability'] = ensembled_result['probability'] + result['probability']*result_ratios[model]
    print(ensembled_result.head(3))

# f = open('lsz/merge/{}_{}_{}_{}_{}_resulta_round2.csv'.format(result_ratios.keys()[0],result_ratios.keys()[1],
#                                                               result_ratios.keys()[2],result_ratios.keys()[3],
#                                                               result_ratios.keys()[4]
#                                                               ), 'w')
# f = open('lsz/merge/{}_{}_{}_resulta_round2.csv'.format(result_ratios.keys()[0],result_ratios.keys()[1],"0131"
#                                                               ), 'w')
# f = open('self_merge/{}_self_merge.csv'.format(model_name
#                                                               ), 'w')
ensembled_result['type'] = ensembled_result['probability'].apply(lambda x:np.argmax(x))
ensembled_result['probability'] = ensembled_result['probability'].apply(lambda x: np2str(x))
print(ensembled_result.head(10))
ensembled_result.to_csv("result/model_merge/{}_prob.csv".format(model_name),index=False)
ensembled_result[['FileName','type']].to_csv("result/model_merge/{}.csv".format(model_name),index=False,header=None)
#
# for idx,probability in enumerate(ensembled_result["probability"].tolist()):
#     one_hot = np.zeros(9, dtype=np.int8)
#     label =np.argmax(probability)
#     one_hot[int(label)] =1
#     one_hot_str =",".join([str(x) for x in list(one_hot)])
#     f.write('%s %s\n' % (doit(str(idx+1)),label+1))
#
# f.close()