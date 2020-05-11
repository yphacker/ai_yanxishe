# 111
[竞赛链接](https://god.yanxishe.com/54)
## score
|model|score|note|
|:---:|:---:|:---:|
|wide_resnet50_2||0.8249|
|densenet201|||


## script
nohup python main.py -m='densenet' -b=64 -e=100 -mode=2 > nohup/densenet.out 2>&1 &

python main.py -o=predict -m='densenet' -b=100