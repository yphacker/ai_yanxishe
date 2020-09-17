# 垃圾分类识别
## data
[链接](https://god.yanxishe.com/84)
## score
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
||0.8839|88.60|0.8788,0.8933,0.8823,0.8817,0.8835|

单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|densenet201|92.41|93.60|epoch=9不在提升了|



## script
nohup python main.py -m='densenet201' -b=32 -e=16 -mode=2 > nohup/densenet201.out 2>&1 &
python main.py -o=predict -m='densenet201'

nohup python main.py -m='densenet201' -b=32 -e=16 > nohup/densenet201.out 2>&1 &


nohup python main.py -m='efficientnet' -b=10 -e=16 -mode=2 > nohup/efficientnet.out 2>&1 &
python main.py -o=predict -m='efficientnet'
nohup python main.py -m='efficientnet' -b=10 -e=16 > nohup/efficientnet.out 2>&1 &
python predict.py -m='efficientnet'
