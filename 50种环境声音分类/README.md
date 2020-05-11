### 50种环境声音分类
[竞赛链接](https://god.yanxishe.com/37)
## score
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|efficientnet|-1|-1|0.9895,0.9912,0.9899,0.9888,0.9908|

单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|efficientnet|87.12|87.5|epoch=100就不再提升了|

## script
nohup python main.py -m='efficientnet' -b=16 -e=100 -mode=2 > nohup/efficientnet.out 2>&1 &
nohup python main.py -m='efficientnet' -b=16 -e=16 > nohup/efficientnet.out 2>&1 &

python main.py -o=predict -m=efficientnet -b=16