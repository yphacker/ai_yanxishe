# coding=utf-8
# author=yphacker

import os
from mrcnn.config import Config

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
model_path = os.path.join(data_path, "model")
model_submission_path = os.path.join(data_path, 'submission.csv')

image_train_path = os.path.join(data_path, 'train')
image_test_path = os.path.join(data_path, 'test')
train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')

epochs_num = 8

mrcnn_model_path = os.path.join(data_path, 'mask_rcnn_coco.h5')
if not os.path.exists(model_path):
    os.makedirs(model_path)
keras_model_dir = os.path.join(model_path, 'model.h5')


# define a configuration for the model， 定义的参数可自行调整
class TrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = "SafetyHelmet_cfg"
    # Number of classes (background + hat + person)
    NUM_CLASSES = 1 + 2
    # Number of training steps per epoch
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3
    STEPS_PER_EPOCH = 50


class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "SafetyHelmet_cfg"
    # Number of classes (background + hat + person)
    NUM_CLASSES = 1 + 2
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
