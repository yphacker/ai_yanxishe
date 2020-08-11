# 美食识别挑战（1）：豆腐VS土豆
## data
[链接](https://god.yanxishe.com/16)
## score
|model|score|note|
|:---:|:---:|:---:|
|resnet|||
|senet|||
|efficientnet-b4|||
|efficientnet-b7|||
|densenet121|||
|inceptionv4|||

## script
nohup python main.py -m='resnet' -b=64 -e=16 -mode=2 > nohup/resnet.out 2>&1 &