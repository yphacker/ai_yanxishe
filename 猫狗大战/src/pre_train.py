# coding=utf-8
# author=yphacker

import os
import shutil
from conf import config


def preprocess_data(mode):
    path = os.path.join(config.data_path, mode)
    data_file = os.listdir(path)  # 读取所有图片的名字
    print(len(data_file))  # 查看数据大小
    # 将图片名为cat和dog的图片分别取出来，存为两个list
    cat_file = list(filter(lambda x: x[:3] == 'cat', data_file))
    dog_file = list(filter(lambda x: x[:3] == 'dog', data_file))
    print(len(dog_file), len(cat_file))

    # 新建文件夹
    for i in ['dog', 'cat']:
        try:
            os.makedirs(os.path.join(config.process_path, mode, i))
        except FileExistsError as e:
            pass

    mode_path = os.path.join(config.process_path, mode)
    obj_path = os.path.join(mode_path, 'cat')
    for i in range(len(cat_file)):
        ori_path = os.path.join(path, cat_file[i])
        shutil.move(ori_path, obj_path)

    obj_path = os.path.join(mode_path, 'dog')
    for i in range(len(dog_file)):
        ori_path = os.path.join(path, dog_file[i])
        shutil.move(ori_path, obj_path)


if __name__ == '__main__':
    preprocess_data('train')
    preprocess_data('val')
