# 英文文本语义相似度
[竞赛链接](https://god.yanxishe.com/53)
## 评估标准
pearson_corr和spearman_corr的平均值
## score:
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert(bert-base-uncased)|-1|-1|0.9895,0.9912,0.9899,0.9888,0.9908|
|bart(bart-large-cnn)|||单折|

单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert(bert-base-uncased)|87.12|-1|epoch=4就不再提升了|
|bart(bart-large-cnn)|92.31|92.6969|b=2, epoch=9就不再提升了|


## script
nohup python main.py -m='bert' -b=16 -e=16 -mode=2 > nohup/bert.out 2>&1 &  
nohup python main.py -m='bert' -b=16 -e=4 > nohup/bert.out 2>&1 &  
nohup python main.py -m='bart' -b=2 -e=16 -mode=2 > nohup/bart.out 2>&1 &  

nohup python main.py -m='bart' -b=2 -e=10 > nohup/bart.out 2>&1 &  

python main.py -o=predict -m=bart -b=2