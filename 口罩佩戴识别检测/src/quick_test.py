
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : fuck.py
#   Author      : YunYang1994
#   Created date: 2019-01-23 10:21:50
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
import  cv2 as cv
from PIL import ImageFont
from PIL import Image
import matplotlib.pyplot as plt




IMAGE_H, IMAGE_W = 416, 416
classes = utils.read_coco_names('./data/mask.names')
num_classes = len(classes)
image_path = "./data/images/"

cpu_nms_graph = tf.Graph()

input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./checkpoint/yolov3_cpu_nms.pb",
                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])



import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config = config)

with tf.Session(graph=cpu_nms_graph) as sess:
    num = 0
    file_handle = open('1.csv', mode='w')
    for i in range(1592):
        path = image_path + str(i) + ".jpg"
        print(i)

        img = Image.open(path)
        img_resized = np.array(img.resize(size=(IMAGE_W, IMAGE_H)), dtype=np.float32)
        # img_resized = np.array(img.reshape())
        img_resized = img_resized / 255.
        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})

        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.45, iou_thresh=0.4)
        if labels is None:
            file_handle.write(str(i) + ',' + str(0) + ',' + str(0))
            file_handle.write('\n')
            num += 1
            print("---------------------------------")
        else:
            list1 = list(labels)
            file_handle.write(str(i)+','+str(list1.count(0))+','+str(list1.count(1)))
            file_handle.write('\n')

    file_handle.close()
    print(num)


