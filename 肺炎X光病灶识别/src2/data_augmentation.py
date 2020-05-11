import math
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import numbers
import types
import collections
import warnings


# 固定角度随机旋转
class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)


def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num-1)
    return img.rotate(angles[index])

