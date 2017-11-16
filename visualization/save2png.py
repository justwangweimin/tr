#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : save2png.py
# @Author: zjj421
# @Date  : 17-11-14
# @Desc  :
import os

import cv2
import numpy as np
from tsahelper.tsahelper import read_data, convert_to_grayscale


def singlefile_save2png(path, dirname1, dirname2, dst_size):
    id = os.path.splitext(os.path.basename(path))[0]
    dirname1 = os.path.join(dirname1, id)
    dirname2 = os.path.join(dirname2, id)
    if not os.path.exists(dirname1):
        os.makedirs(dirname1)
    if not os.path.exists(dirname2):
        os.makedirs(dirname2)
    # read in the aps file, it comes in as shape(512, 620, 16)
    imgs = read_data(path)
    # transpose so that the slice is the first dimension shape(16, 620, 512)
    imgs = imgs.transpose()
    for i, img in enumerate(imgs):
        path_1 = os.path.join(dirname1, "src_{id}_{i}.png".format(id=id, i=i))
        img = convert_to_grayscale(img)
        # correct the orientation of the image
        img = np.flipud(img)
        cv2.imwrite(path_1, img)
        # 图片缩放
        resized_img = cv2.resize(img, dst_size, interpolation=cv2.INTER_AREA)
        # 灰度图片转RGB
        new_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
        path_2 = os.path.join(dirname2, "dst_{id}_{i}.png".format(id=id, i=i))
        cv2.imwrite(path_2, new_img)
    print('Done!')


# 把指定目录(data_root)中的所有文件保存为png文件
# 如果指定了single_path, 则把指定文件保存为png文件
def save2png(save_root, data_root, dst_size, single_path=None):
    root_ = save_root
    dir_1 = "src_imgs"
    dirname1 = os.path.join(root_, dir_1)
    dir_2 = "dst_imgs"
    dirname2 = os.path.join(root_, dir_2)
    if not single_path:
        path_list = []
        for root, dirs, files in os.walk(data_root):
            for file in files:
                path = os.path.join(root, file)
                path_list.append(path)
        for path in path_list:
            singlefile_save2png(path, dirname1, dirname2, dst_size)
    else:
        singlefile_save2png(single_path, dirname1, dirname2, dst_size)


def __main():
    save2png(SAVE_ROOT, DATA_ROOT, DST_SIZE, SINGLE_PATH)

if __name__ == '__main__':
    SAVE_ROOT = "/home/zj/桌面/a3daps"
    DATA_ROOT = None
    DST_SIZE = (200, 200)
    SINGLE_PATH = "/home/zj/helloworld/kaggle/tr/input/sample/00360f79fd6e02781457eda48f85da90.a3daps"
    __main()
