# coding=utf-8
# author=yphacker

import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")

train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "val")
test_path = os.path.join(data_path, "test")

process_path = os.path.join(data_path, "process")
process_train_path = os.path.join(process_path, "train")
process_val_path = os.path.join(process_path, "val")
model_path = os.path.join(data_path, "model")
model_save_path = os.path.join(model_path, "resnet18.bin")

image_size = 224
batch_size = 128
epochs_num = 1
print_per_batch = 10
