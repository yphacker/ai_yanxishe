# 美食识别挑战（2）：茄子、山药、苦瓜、西兰花
## data
[链接](https://god.yanxishe.com/26)
## score
|model|score|note|
|:---:|:---:|:---:|


## script
nohup python main.py -m='resnet' -b=64 -e=16 > nohup/resnet.out 2>&1 &

nohup python main.py -m='resnet' -b=32 -e=16 > nohup/resnet.out 2>&1 &