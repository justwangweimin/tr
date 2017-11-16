#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : a3dapsfile.py
# @Author: zjj421
# @Date  : 17-11-14
# @Desc  :
import cv2
from tsahelper.tsahelper import read_data, convert_to_grayscale
import matplotlib.pyplot as plt
from offline.functions.imgs_preprocessed.img_preprocessed_save2hdf5 import preprocess_imgs_per_subject
from online.mymodule.preprocess_data import my_read_data
import numpy as np


def plot_image_set1(infile):
    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)

    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()

    # show the graphs
    fig, axarr = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))

    i = 0
    for row in range(8):
        for col in range(8):
            # print(img[i].shape)
            img[i] = convert_to_grayscale(img[i])
            cv2.imwrite("/home/zj/桌面/testa3daps_{}.png".format(i), np.flipud(img[i]))
            resized_img = cv2.resize(img[i], TARGET_IMG_SIZE, fx=0.1, fy=0.1)
            # print(resized_img.shape)
            axarr[row, col].imshow(np.flipud(resized_img), cmap=COLORMAP)
            resized_img = convert_to_grayscale(resized_img)
            cv2.imwrite("/home/zj/桌面/testa3dapsresized_{}.png".format(i), np.flipud(resized_img))
            i += 1
    plt.show()
    print('Done!')


def __main():
    plot_image_set1(PATH)
    # imgs_per_subject = my_read_data(PATH)
    # print(type(imgs_per_subject))
    # print("@@"*100)
    # new_imgs = preprocess_imgs_per_subject(imgs_per_subject, TARGET_IMG_SIZE)


if __name__ == '__main__':
    TARGET_IMG_SIZE = (200, 200)
    COLORMAP = 'pink'

    PATH = "/home/zj/helloworld/kaggle/tr/input/sample/0043db5e8c819bffc15261b1f1ac5e42.a3daps"
    __main()
