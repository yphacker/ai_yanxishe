# 真假职位信息检测
[竞赛链接](https://god.yanxishe.com/46)
## score
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert(bert-base-uncased)|-1|-1|-1|

单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|CatBoostClassifier|-1|94|
|bert(bert-base-uncased)|98.87|89|epoch=?就不再提升了|


## script
nohup python main.py -m='bert' -b=32 -e=16 -mode=2 > nohup/bert.out 2>&1 &  
nohup python main.py -m='bert' -b=16 -e=3 > nohup/bert.out 2>&1 &  

python main.py -o=predict -m=bert -b=32
