# 美食识别挑战（2）：茄子、山药、苦瓜、西兰花
## data
[链接](https://god.yanxishe.com/26)
## score
|model|score|note|
|:---:|:---:|:---:|
|resnet|98.1308||
|senet|98.4813||
|efficientnet-b4|97.5467||
|efficientnet-b7|97.8972||
|densenet121|97.8972||
|inceptionv4|98.014||
|resnet+senet+efficientnet|98.5981|2,3,1|
|resnet+senet+efficientnet+densenet121+inceptionv4|98.5981|3,4,1,1,2|
|resnet+senet+efficientnet-b4+efficientnet+densenet121+inceptionv4|98.5981|3,3,1,1,1,1|


## script
nohup python main.py -m='resnet' -b=64 -e=16 > nohup/resnet.out 2>&1 &