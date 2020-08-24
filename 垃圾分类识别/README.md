# 垃圾分类识别
## data
[链接](https://god.yanxishe.com/84)
## score
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|resnet18|0.9456|96.83|0.9435,0.9457,0.9480,0.9305,0.9602|

|model|score|note|
|:---:|:---:|:---:|
|efficientnet-b5|0.9604|epoch=13不在提升了|


## script
nohup python main.py -m='densenet201' -b=10 -e=16 -mode=2 > nohup/efficientnet.out 2>&1 &

nohup python main.py -m='efficientnet' -b=10 -e=16 -mode=2 > nohup/efficientnet.out 2>&1 &
python main.py -o=predict -m='efficientnet'
nohup python main.py -m='efficientnet' -b=10 -e=16 > nohup/efficientnet.out 2>&1 &
python predict.py -m='efficientnet'
