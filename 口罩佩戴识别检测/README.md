### 口罩佩戴识别检测
## 数据下载
[数据下载](https://god.yanxishe.com/38)

训练步骤如下
step1：下载好yolov3.weights权重文件，将其放入checkpoint文件夹下
step2：获得训练数据集格式如下
xxx/xxx.jpg 18.19 6.32 424.13 421.83 20 323.86 2.65 640.0 421.94 20 
# image_path x_min y_min x_max y_max class_id  x_min y_min ... class_id 
在这里我将佩戴口罩class_id设为1，没有佩戴口罩class_id设置为0
step3：class.names文件编写 我们编写好了两类的文件为mask.names文件

step4：通过运行kmeans.py代码获得mask_anchors.txt格式如下：
0.10,0.13, 0.16,0.30,0.33,0.23, 0.40,0.61,0.62,0.45, 0.69,0.59, 0.76,0.60, 0.86,0.68, 0.91,0.76
我们得到的是mask_anchors.txt 文件
step5：运行如下命令在train_data_tfrecord文件夹下得到.tfrecord文件，并命名为train.tfrecord和test.tfrecord
python convert_tfrecord.py --dataset_txt GroundTruth.txt --tfrecord_path_prefix ../data/train_data_tfrecord/

step6：通过运行train.py进行训练
python convert_weight.py -cf ./checkpoint/yolov3.ckpt-92000 -nc 2  -ap ./data/mask_anchors.txt --freeze
setp7：通过train.py获得训练好的权重文件,之后用如下命令转为yolov3_cpu_nms.pb文件


(转换好的yolov3_cpu_nms.pb文件在checkpoint文件夹下，用于quick_test.py调用)

测试步骤
运行quick_test.py生成1.csv最终结果文件，选取合适的阈值最终得分为95.2261