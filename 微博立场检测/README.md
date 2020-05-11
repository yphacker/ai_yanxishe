# 微博立场检测
[竞赛链接](https://god.yanxishe.com/44)
## 评估标准
acc
## kaggle score:
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert(chinese_roberta_wwm_ext)|69.58|70.5|epoch=6就不再提升了|
|bert(chinese_roberta_wwm_ext)|0.6691|0.7216|0.6715,0.6798,0.6854,0.6625,0.6464|




## script
nohup python main.py -m='bert' -b=32 -e=16 -mode=2 > nohup/bert.out 2>&1 &  
nohup python main.py -m='bert' -b=32 -e=8 > nohup/bert.out 2>&1 &  

python main.py -o=predict -m=bert
python predict.py -m=bert