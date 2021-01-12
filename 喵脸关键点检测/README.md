# 喵脸关键点检测
## data
[链接](https://god.yanxishe.com/19)
## score
单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|densenet201|5.2646||epoch=30不在提升了|


5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|

## script
nohup python main.py -m='densenet201' -b=32 -e=100 -mode=2 > nohup/densenet201.out 2>&1 &
python main.py -o=predict -m='densenet201'

nohup python main.py -m='densenet201' -b=32 -e=36 > nohup/densenet201.out 2>&1 &



