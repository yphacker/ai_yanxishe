# 看画猜作者
## data
[链接](https://god.yanxishe.com/90)
## score
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|densenet201|0.8839|88.60|0.8788,0.8933,0.8823,0.8817,0.8835|

单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|densenet201|81.05||epoch=36不在提升了|



## script
nohup python main.py -m='densenet201' -b=32 -e=100 -mode=2 > nohup/densenet201.out 2>&1 &
python main.py -o=predict -m='densenet201'

nohup python main.py -m='densenet201' -b=32 -e=36 > nohup/densenet201.out 2>&1 &



