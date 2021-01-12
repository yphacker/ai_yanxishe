# 国外名人识别
## data
[链接](https://god.yanxishe.com/83)
## score
### 单折
|model|score|note|
|:---:|:---:|:---:|
|densenet201|89.99|||

### 5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|densenet201||||

## script
nohup python main.py -m='densenet201' -b=32 -e=16 -mode=2 > nohup/densenet201.out 2>&1 &
python main.py -m='densenet201' -o=predict

nohup python main.py -m='densenet201' -b=32 -e=5 > nohup/densenet201.out 2>&1 &
python predict.py -m='densenet201'

