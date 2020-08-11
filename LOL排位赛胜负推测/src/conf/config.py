# coding=utf-8
# author=yphacker


import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
orig_data_path = os.path.join(data_path, "origin")
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")
