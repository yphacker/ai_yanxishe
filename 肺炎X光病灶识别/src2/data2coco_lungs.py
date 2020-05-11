# -*- coding: utf-8 -*-
'''
@time: 2018/10/12 19:22
数据的标注非常坑，json里面的imagePath和真实的图片名称对不上，因而采用替换.json为.jpg的方法
@ author: javis
'''

import os
import json
import numpy as np
import glob
import cv2
import shutil
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#0为背景

defect_name2label = {
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15
}

box_length_list=[]
box_scale_list=[]

class Lableme2CoCo:

    def __init__(self,mode="train",img_root=None,add_vflip=False):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode=mode
        self.add_vflip=add_vflip
        self.img_root=img_root
        if not os.path.exists("../lungs_hflip/images/{}".format(self.mode)):
            os.makedirs("../lungs_hflip/images/{}".format(self.mode))
    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示
        # json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, anno_pd,):
        self._init_categories()
        head_sum=0
        for img_name,bbox in tqdm.tqdm(zip(anno_pd["ID"].tolist(),anno_pd["bbox"].tolist())):
            # ori
            # info = info.replace("\n","").split(" ")
            img_path = os.path.join(self.img_root,img_name)

            self._cp_img(img_path)
            head_num =len(bbox.split(";"))
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            self.images.append(self._image(img_path,h, w))
            if int(head_num)==0:
                continue
            head_sum +=int(head_num)

            for annotation in bbox.split(";"):
                annotation=annotation.split(" ")

                annotation = list(map(float, annotation))
                annotation = list(map(abs, annotation))

                # img =cv2.rectangle(img,(annotation[0],annotation[1]),(annotation[2],annotation[3]),(255,0,255),1)
                annotation = self._annotation(annotation,img_path)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
            # plt.imshow(img)
            # plt.show()

            #add vflip
            if self.add_vflip:
                self._save_vflip_img(img,img_path)
                self.images.append(self._image(img_path,h, w,vflip=True))
                for annotation in bbox.split(";"):
                    annotation = annotation.split(" ")
                    annotation = list(map(float, annotation))
                    annotation = list(map(abs, annotation))

                    annotation = self._annotation(annotation,w,vflip=True)
                    self.annotations.append(annotation)
                    self.ann_id += 1
                self.img_id += 1
        print("head sum",head_sum)
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance


    # 构建类别
    def _init_categories(self):
        for v in range(1,3):
            category = {}
            category['id'] = v
            category['name'] = str(v)
            category['supercategory'] = 'defect_name'
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path,h,w,vflip=False):
        image = {}
        # img=cv2.imread(path)
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        if not vflip:
            image['file_name'] = os.path.basename(path)
        else:
            image['file_name'] = os.path.basename(path).replace(".jpg", "_vflip.jpg")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, points,w,vflip=False):
        label = points[4]
        area=(points[2]-points[0])*(points[3]-points[1])
        box_length_list.append(max(points[2]-points[0],points[3]-points[1]))
        box_scale_list.append(1.0*max(points[2]-points[0],points[3]-points[1])/min(points[2]-points[0],points[3]-points[1]))
        points=[[points[0],points[1]],[points[2],points[1]],[points[2],points[3]],[points[0],points[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(label)
        if vflip:
            for indx in range(len(points)):
                points[indx][0]=w-points[indx][0]-1


        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)

        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        shutil.copy(img_path, os.path.join("../lungs_hflip/images/{}".format(self.mode), os.path.basename(img_path)))

    def _save_vflip_img(self, img,img_path):
        # img = cv2.imread(img_path)
        img_flip = cv2.flip(img, 1)
        cv2.imwrite(os.path.join("../lungs_hflip/images/{}".format(self.mode), os.path.basename(img_path)).replace(".jpg", "_vflip.jpg"),img_flip)
    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]
if __name__ == '__main__':
    import os
    # 把所有的jpg和json目录
    base_dir = '/'
    img_info_list=pd.read_csv("/media/hszc/data/datafountain/lungs/code/train.csv")

    print(img_info_list.head())


    l2c_train = Lableme2CoCo(mode="train2017",img_root=base_dir,add_vflip=True)
    train_instance = l2c_train.to_coco(img_info_list)
    if not os.path.exists("../lungs_hflip/{}".format("annotations")):
        os.makedirs("../lungs_hflip/{}".format("annotations"))
    # plt.subplot(211)
    # box_length_list.sort()
    # box_scale_list.sort()
    # plt.plot(box_length_list)
    # plt.subplot(212)
    # plt.plot(box_scale_list)
    #
    # plt.show()
    l2c_train.save_coco_json(train_instance, "../lungs_hflip/{}/".format("annotations")+'instances_train2017.json')

    # # 把验证集转化为COCO的json格式
    # l2c_val = Lableme2CoCo(mode="val2017",img_root=base_dir)
    # val_instance = l2c_val.to_coco(val_pd)
    # l2c_val.save_coco_json(val_instance, "coco/{}/".format("annotations")+'instances_val2017.json')