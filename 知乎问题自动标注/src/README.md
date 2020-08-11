# 
## data
[链接](https://god.yanxishe.com/75)
## score
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert(bert-base-chinese)|0.8839|88.60|0.8788,0.8933,0.8823,0.8817,0.8835|

单折
|model|online score|note|
|:---:|:---:|:---:|
|bert(bert-base-chinese)|0.2440|第8轮loss还在降低|


## script
nohup python main.py -m='bert' -b=32 -e=8 -mode=2 > nohup/bert.out 2>&1 &  
python main.py -o=predict -m=bert -b=100
nohup python main.py -m='bert' -b=32 -e=4 -mode=2 > nohup/bert.out 2>&1 &  
