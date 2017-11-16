#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : img_preprocessed_save2hdf5.py
# @Author: zjj421
# @Date  : 17-10-8
# @Desc  :
import os
from datetime import datetime

import cv2
import h5py
import numpy as np
from tsahelper.tsahelper import convert_to_grayscale, get_subject_labels, get_subject_zone_label

from online.mymodule.data_random_classfication import get_label_data_list
from online.mymodule.preprocess_data import my_read_data


def preprocess_imgs_per_subject(imgs, target_img_size):
    target_img_size = target_img_size
    new_imgs = []
    for img in imgs:
        # correct the orientation of the image
        new_img = np.flipud(img)
        # scale pixel values to grayscale
        # new_img = convert_to_grayscale(new_img)
        # 图片缩放
        new_img = cv2.resize(new_img, target_img_size, interpolation=cv2.INTER_AREA)  # return a new picture
        print(new_img.shape)
        # 灰度图片转RGB
        new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
        print(new_img.shape)
        new_imgs.append(new_img)
    return new_imgs


def get_single_label(stage1_labels, subject_id):
    num_zones = 17  # 17个区块
    # 读取某个id的17个区块的标签值
    df = get_subject_labels(stage1_labels, subject_id)
    zone_label_list = []
    for zone_nth in range(num_zones):
        # 读取某个id某个区块的标签值
        yes_or_no = get_subject_zone_label(zone_nth, df)
        # print(yes_or_no)
        if yes_or_no == [1, 0]:
            yes_or_no = 0
        else:
            yes_or_no = 1
        zone_label_list.append(yes_or_no)
    return zone_label_list

# 把预处理过的图片与对应的标签保存为h5文件。
def save_imgs_preprocessed_labels2hdf5(data_root, labels_group, stage1_labels, imgs_trained_group, imgs_tested_group, target_img_size):
    label_data_list = get_label_data_list(data_root, stage1_labels)
    for root, dirs, files in os.walk(data_root):
        for i, file in enumerate(files):
            subject_id = os.path.splitext(file)[0]
            if subject_id in label_data_list:
                zone_label_list = get_single_label(stage1_labels, subject_id)
                # 每个dataset包含一个人的一组标签
                # 默认压缩等级为４
                labels_group.create_dataset(str(subject_id), data=zone_label_list, compression="gzip",
                                            compression_opts=4)
                imgs_group = imgs_trained_group
            else:
                imgs_group = imgs_tested_group

            # 读取图片，每个aps文件包含16张图片。
            path = os.path.join(root, file)
            imgs_per_subject = my_read_data(path)
            new_imgs = preprocess_imgs_per_subject(imgs_per_subject, target_img_size)
            # 每个dataset包含一个人的16张照片
            imgs_group.create_dataset(str(subject_id), data=new_imgs, compression="gzip", compression_opts=4)
            print("-"*100, i)

    print("Done")


def _main():
    save_imgs_preprocessed_labels2hdf5(DATA_ROOT, LABELS_GROUP, STAGE1_LABELS, IMGS_TRAINED_GROUP, IMGS_TESTED_GROUP,
                                       TARGET_IMG_SIZE)


if __name__ == '__main__':
    TARGET_IMG_SIZE = (200, 200)
    f_name = "imgs_preprocessed_200_200_3.h5"
    if os.path.exists(f_name):
        print("请重新指定hdf5文件名！")
        exit()
    F = h5py.File(f_name, "a")
    DATA_ROOT = "/media/zj/study/kaggle/stage1_aps"
    try:
        IMGS_TRAINED_GROUP = F.create_group("imgs_trained")
    except:
        IMGS_TRAINED_GROUP = F["imgs_trained"]
    try:
        IMGS_TESTED_GROUP = F.create_group("imgs_tested")
    except:
        IMGS_TESTED_GROUP = F["imgs_tested"]
    try:
        LABELS_GROUP = F.create_group("labels")
    except:
        LABELS_GROUP = F["labels"]
    STAGE1_LABELS = "/media/zj/study/kaggle/stage1_labels.csv"

    begin = datetime.now()
    print("开始时间： ", begin)
    _main()
    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))
