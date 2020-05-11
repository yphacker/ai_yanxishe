# 看图识车
## data
[链接](https://god.yanxishe.com/26)
## score
|model|score|note|
|:---:|:---:|:---:|
|resnet|||
|senet|||
|efficientnet-b7|||
|densenet121|||
|inceptionv4|||

## script
nohup python main.py -m='resnet' -b=64 -e=128 -mode=2 > nohup/resnet.out 2>&1 &