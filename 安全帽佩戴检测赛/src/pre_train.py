# coding=utf-8
# author=yphacker

import os
import re
import pandas as pd
from xml.dom.minidom import parse
import xml.dom.minidom
from conf import config


def test():
    dir_path = '../data/label'
    for filename in os.listdir(dir_path):
        print(filename)
        file_path = '{}/{}'.format(dir_path, filename)
        dom = xml.dom.minidom.parse(file_path)
        root = dom.documentElement
        objs = root.getElementsByTagName('object')
        for obj in objs:
            xmin = obj.getElementsByTagName('xmin')[0].childNodes[0].data
            ymin = obj.getElementsByTagName('ymin')[0].childNodes[0].data
            xmax = obj.getElementsByTagName('xmax')[0].childNodes[0].data
            ymax = obj.getElementsByTagName('ymax')[0].childNodes[0].data
            name = obj.getElementsByTagName('name')[0].childNodes[0].data
            print(xmin, ymin, xmax, ymax, name)
        break


def solve():
    img_paths = []
    boxes = []
    dir_path = '../data/label'
    for filename in os.listdir(dir_path):
        # print(filename)
        file_path = '{}/{}'.format(dir_path, filename)
        dom = xml.dom.minidom.parse(file_path)
        root = dom.documentElement
        objs = root.getElementsByTagName('object')
        box = []
        for obj in objs:
            xmin = obj.getElementsByTagName('xmin')[0].childNodes[0].data
            ymin = obj.getElementsByTagName('ymin')[0].childNodes[0].data
            xmax = obj.getElementsByTagName('xmax')[0].childNodes[0].data
            ymax = obj.getElementsByTagName('ymax')[0].childNodes[0].data
            name = obj.getElementsByTagName('name')[0].childNodes[0].data
            # print(xmin, ymin, xmax, ymax, name)
            box.append('{},{},{},{},{}'.format(xmin, ymin, xmax, ymax, 0 if name == 'hat' or name == 'dog' else 1))
        search_obj = re.search('(\d+)', filename)
        if search_obj:
            # print(search_obj.group(1))
            id = search_obj.group(1)
            img_path = '{}/{}.jpg'.format(config.image_train_path, id)
            img_paths.append(img_path)
            boxes.append(' '.join(box))
        else:
            print('{}, id解析不出来'.format(filename))
    df = pd.DataFrame({'img_path': img_paths, 'box': boxes})
    df.to_csv(config.train_path, index=None)


if __name__ == '__main__':
    solve()
