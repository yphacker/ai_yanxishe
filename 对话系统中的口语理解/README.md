# 对话系统中的口语理解
## data
[链接](https://god.yanxishe.com/69)
## score
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|

单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert-base-chinese|90.83|41|epoch=3就不再提升了|


## script
nohup python main.py -m='bert' -b=32 -e=16 -mode=2 > nohup/bert.out 2>&1 &  
python main.py -o=predict -m=bert -b=32  

