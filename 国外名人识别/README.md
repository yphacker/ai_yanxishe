# 人脸检测--国外名人识别
## data
[链接](https://god.yanxishe.com/48)
## score
### 单折
|model|score|note|
|:---:|:---:|:---:|
|densenet201|0.9486|epoch=5就不再提升了|

### 5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|densenet201|0.9945|1|1.0000,0.9955,1.0000,0.9817,0.9954|

## script
nohup python main.py -m='densenet201' -b=50 -e=16 -mode=2 > nohup/densenet201.out 2>&1 &
nohup python main.py -m='densenet201' -b=25 -e=16 -mode=2 > nohup/densenet201.out 2>&1 &
python main.py -m='densenet201' -o=predict

nohup python main.py -m='densenet201' -b=32 -e=5 > nohup/densenet201.out 2>&1 &
python predict.py -m='densenet201'

