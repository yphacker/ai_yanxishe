# 验证码识别大赛（一）
## data
[链接](https://god.yanxishe.com/66)
## score

## Train
```
   [run] python train.py --cfg lib/config/360CC_config.yaml
or [run] python train.py --cfg lib/config/OWN_config.yaml
```
```
#### loss curve

```
   [run] cd output/360CC/crnn/xxxx-xx-xx-xx-xx/
   [run] tensorboard --logdir log
```

## Demo
```
   [run] python demo.py 
```
## References
- https://github.com/meijieru/crnn.pytorch
- https://github.com/HRNet
- https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec

nohup python train.py --cfg lib/config/OWN_config.yaml > info.out 2>&1 &  



