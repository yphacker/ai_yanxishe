import tensorflow as tf
import numpy as np


def flip(x):
    '''
    Flip augmentation

    :param x: Image to flip
    :return: Augmented image

    '''
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


def color(x):
    '''
    color agumentation

    :param x: Image
    :return: Agumented image

    '''
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)

    return x


def rotate(x):
    '''
    Zoom augmentation

    :param x: Image
    :return: Augmented image

    '''
    x = tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    return x


# 使用了常规的数据扩增方法，分别对30%的输入数据进行水平翻转、颜色增强、旋转以及缩放。
def zoom(x):
    '''
    Zoom augmention

    :param x: Image
    :return: Augmented image

    '''
    # Generate 20 crop settings, ranging from 1% to 20% crop
    scales = list(np.arange(0.8, 1, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scales in enumerate(scales):
        x1 = y1 = 0.3 - (0.3 * scales)
        x2 = y2 = 0.3 + (0.3 + scales)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(20), crop_size=(128, 128))

        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=20, dtype=tf.int32)]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
    return x
