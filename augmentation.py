import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def vertical_shift_down(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    img = fill(img, h, w)
    return img

def vertical_shift_up(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(0.0, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    img = fill(img, h, w)
    return img

def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

def create_augment(aug, dir, images, target, dsize):
    for index, fname in enumerate(images):
        print(fname)
        i = 0
        while (i < aug):
            img = cv2.imread(fname)
            img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            img = horizontal_shift(img, 0.1)
            img = vertical_shift_down(img, 0.1)
            img = vertical_shift_up(img, 0.1)
            file_name = "%s/%s_%s_%s" % (dir, index, i, str(target[index]) + '.jpg')
            cv2.imwrite(file_name, img)
            i = i + 1

aug = 100

dir = 'augmentation/0/'
images = sorted(glob.glob('result/0/*.jpg'))
target = [0,1,0,0,0]
dsize = (40, 135)

create_augment(aug, dir, images, target, dsize)

dir = 'augmentation/1/'
images = sorted(glob.glob('result/1/*.jpg'))
target = [0,0,0,0,0,0,1,0,1,0]
dsize = (50, 135)

create_augment(aug, dir, images, target, dsize)

dir = 'augmentation/2/'
images = sorted(glob.glob('result/2/*.jpg'))
target = [1,2,1,0,0,0]
dsize = (100, 100)

create_augment(aug, dir, images, target, dsize)

dir = 'augmentation/3/'
images = sorted(glob.glob('result/3/*.jpg'))
target = [1,0,1,0,1,0,1,0]
dsize = (100, 155)

create_augment(aug, dir, images, target, dsize)
