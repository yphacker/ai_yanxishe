1、尝试使用过Cascade RCNN检测模型，发现效果很差，只有78分。可视化图片后发现标记bbox的位置非常粗糙，后来选择了分类模型。
2、查看数据分布发现病灶数量3和4的样本数量极少，所以把其当做2的类别训练三分类模型。
3、图片中语义信息较少，纹理信息较多，所以采用SeNet154和efficientnet-b7大网络进行融合。
4、create_train_5flod.py划分训练集验证集
5、create_test.py制作测试集
6、train_main.py训练模型并预测
7、self_merge.py五折模型自融合
8、model_merge.py模型之间融合