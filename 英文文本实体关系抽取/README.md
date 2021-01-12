# 英文文本实体关系抽取
## data
[链接](https://god.yanxishe.com/82)
## score
### 5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|densenet201|0.9945|1|1.0000,0.9955,1.0000,0.9817,0.9954|

### 单折
|model|score|note|
|:---:|:---:|:---:|
|densenet201|0.9486|epoch=11就不再提升了|

## script
nohup python main.py -m='bert' -b=128 -e=8 -mode=2 > nohup/bert.out 2>&1 &  
python main.py -o=predict -m=bert

nohup python main.py -m='bert' -b=32 -e=4 > nohup/bert.out 2>&1 & 

