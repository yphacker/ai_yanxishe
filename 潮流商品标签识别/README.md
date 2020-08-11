# 潮流商品标签识别
## data
[链接](https://god.yanxishe.com/74)
## score
|model|score|note|
|:---:|:---:|:---:|

|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|resnet18|0.9817|99.27|0.9820,0.9864,0.9771,0.9771,0.9862|
|wide_resnet50_2|0.9936|1|0.9910,1.0000,0.9954,0.9908,0.9908|
|densenet121|0.9900|99.27|0.9865,0.9955,0.9954,0.9817,0.9908|
|densenet201|0.9945|1|1.0000,0.9955,1.0000,0.9817,0.9954|

## script
nohup python main.py -m='resnet18' -b=64 -e=16 > nohup/resnet18.out 2>&1 &
nohup python main.py -m='wide_resnet50_2' -b=64 -e=16 > nohup/wide_resnet50_2.out 2>&1 &
nohup python main.py -m='densenet121' -b=64 -e=16 > nohup/densenet121.out 2>&1 &
nohup python main.py -m='densenet201' -b=32 -e=16 > nohup/densenet201.out 2>&1 &

python predict.py -m='resnet18'
python predict.py -m='wide_resnet50_2'
python predict.py -m='densenet121'
python predict.py -m='densenet201'

