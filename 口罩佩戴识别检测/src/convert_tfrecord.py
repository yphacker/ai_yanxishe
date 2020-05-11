#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_tfrecord.py
#   Author      : YunYang1994
#   Created date: 2018-12-18 12:34:23
#   Description :
#
#================================================================
#todo: python convert_tfrecord.py  --dataset_txt  plate_train_20000.txt  --tfrecord_path_prefix  ./train_data_tfrecord/
import sys
import argparse
import numpy as np
import tensorflow as tf

def main(argv):
    parser = argparse.ArgumentParser() #命令行参数
    parser.add_argument("--dataset_txt", default=None)# 数据文件标签文件
    parser.add_argument("--tfrecord_path_prefix", default=None)#转成TFrecord文件路径
    flags = parser.parse_args()#建立索引
    dataset = {} #建立空的字典
    with open(flags.dataset_txt,'r') as f:#打开标签文件
        for line in f.readlines():# 读取每一行信息
            example = line.split(' ')# 按照空格进行分割
            #print(type(example))
            image_path = example[0] #图片路径
            print(image_path)
            boxes_num = len(example[1:]) // 5 #根据长度分析每一行标签中包含了几个目标框
            boxes = np.zeros([boxes_num, 5], dtype=np.float32)#新建格式为[boxes_num, 5]的矩阵用来存放每一个框的标签数据
            for i in range(boxes_num):
                boxes[i] = example[1+i*5:6+i*5]#将标签文件数据存放在boxes当中
            dataset[image_path] = boxes#路径图片和标签一一对应
    image_paths = list(dataset.keys())#将所有的图片路径列出来
    images_num = len(image_paths)
    print(">> Processing %d images" %images_num)
    #总共右多少张图片

    tfrecord_file = flags.tfrecord_path_prefix+".tfrecords"#tfrecord文件路径
    with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:#建立读写器
        for i in range(images_num):#循环遍历每一个图片
            image = tf.gfile.FastGFile(image_paths[i], 'rb').read()#读取图片
            boxes = dataset[image_paths[i]]
            boxes = boxes.tostring()

            example = tf.train.Example(features = tf.train.Features(
                feature={
                    'image' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                    'boxes' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [boxes])),
                }
            ))

            record_writer.write(example.SerializeToString())
        print(">> Saving %d images in %s" %(images_num, tfrecord_file))


if __name__ == "__main__":
    main(sys.argv[1:])