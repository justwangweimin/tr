#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : preprocess_data.py
# @Author: zjj421
# @Date  : 17-9-11
# @Desc  :
import os

import cv2
import numpy as np
from tsahelper.tsahelper import get_subject_labels, get_subject_zone_label, read_data, convert_to_grayscale


def my_read_data(file):
    imgs = read_data(file)
    print("原始16个不同视图的图片的shape: {}".format(imgs.shape))
    # ----------------------------------------------------
    # 图片的shape（高，宽）与（宽，高）对图片本身来说是否一样？？
    # (512,660,16) --> (16, 660, 512)
    imgs = imgs.transpose()
    print("转换过的不同视图的图片的shape: {}".format(imgs.shape))
    return imgs


def get_lst_all_imgs_and_ids(data_root):
    lst_imgs_per_subject = []
    # 每个subject分为17个区块，id由subject_id+区块名组成
    subject_id_set = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            filename_path = os.path.join(root, file)
            # 读取图片，每个aps文件包含16张图片。
            imgs_per_subject = my_read_data(filename_path)
            lst_imgs_per_subject.append(imgs_per_subject)
            subject_id = os.path.splitext(file)[0]
            subject_id_set.append(subject_id)

    print("-" * 100)
    print("图片读取结束！")
    print("-" * 100)
    num_subjects = len(lst_imgs_per_subject)
    print("该批次总共有{}人:\n".format(num_subjects))
    num_imgs_per_subject = len(lst_imgs_per_subject[0])  # 16张不同角度的照片
    print("每个人有{}张不同视图的照片".format(num_imgs_per_subject))
    pxh = len(lst_imgs_per_subject[0][0])
    pxw = len(lst_imgs_per_subject[0][0][0])
    print("每张图片的像素为: {}x{}.".format(pxh, pxw))
    return lst_imgs_per_subject, subject_id_set


def preprocess_imgs(lst_imgs_per_subject):
    # target_img_size = (150, 150)
    target_img_size = (200, 200)
    # 处理后的所有图片数据
    lst_new_imgs = []
    # 图片resize和gray2RGB
    for s in lst_imgs_per_subject:
        new_imgs = []
        for img in s:
            # correct the orientation of the image
            new_img = np.flipud(img)
            # scale pixel values to grayscale
            new_img = convert_to_grayscale(new_img)
            # 图片缩放成（150,150）
            new_img = cv2.resize(new_img, target_img_size, interpolation=cv2.INTER_AREA)  # return a new picture
            # 灰度图片转RGB
            new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
            new_imgs.append(new_img)
        lst_new_imgs.append(new_imgs)
    print("preprocess_imgs--所有图片处理完毕！")
    print("-" * 100)
    return lst_new_imgs


def get_labels(subject_id_set_list, stage1_labels):
    num_zones = 17  # 17个区块
    lst_labels = []
    for subject_id in subject_id_set_list:
        df = get_subject_labels(stage1_labels, subject_id)
        zone_label_list = []
        for zone_nth in range(num_zones):
            yes_or_no = get_subject_zone_label(zone_nth, df)
            # print(yes_or_no)
            if yes_or_no == [1, 0]:
                yes_or_no = 0
            else:
                yes_or_no = 1
            zone_label_list.append(yes_or_no)
        lst_labels.append(zone_label_list)
    print("get_labels--Over")
    return lst_labels


def zip_train_data(x_train, y_train):
    train_data = []
    for i in zip(x_train, y_train):
        train_data.append(i)
    return train_data


# (None, 16, height, weight, 3)
# 每个人的16张照片对应一组标签
def preprocess_data_1(data_root, stage1_labels=None):
    lst_imgs_per_subject, subject_id_set_list = get_lst_all_imgs_and_ids(data_root)
    lst_imgs_per_subject = preprocess_imgs(lst_imgs_per_subject)
    if not stage1_labels:
        return lst_imgs_per_subject, subject_id_set_list
    labels_set = get_labels(subject_id_set_list, stage1_labels)
    return lst_imgs_per_subject, labels_set


# flatten所有照片,每张照片对应一组标签，每个人的16张视图的标签一样.
# (None, 16, height, weight, 3) -> (None, height, weight, 3)
def preprocess_data_2(lst_imgs_per_subject, lst_labels_set=None):
    lst_imgs_per_subject = np.asarray(lst_imgs_per_subject, dtype=np.float32)
    # all_imges_array = lst_imgs_per_subject.reshape(-1, 150, 150, 3)
    all_imges_array = lst_imgs_per_subject.reshape(-1, 200, 200, 3)
    num_angles = 16
    if not lst_labels_set:
        return all_imges_array
    labels_list = [x for x in lst_labels_set for i in range(num_angles)]
    labels_list = np.asarray(labels_list, dtype=np.float32)
    return all_imges_array, labels_list


# (None, 16, height, weight, 3) -> (16, None, height, weight, 3)
# 每个人的16张照片对应一组标签
def preprocess_data_3(lst_imgs_per_subject):
    lst_imgs_per_subject = np.asarray(lst_imgs_per_subject, dtype=np.float32)
    num_angles = 16
    all_imgs_list = np.array_split(lst_imgs_per_subject, num_angles, axis=1)
    for i, x in enumerate(all_imgs_list):
        # x = x.reshape(-1, 150, 150, 3)
        x = x.reshape(-1, 200, 200, 3)
        all_imgs_list[i] = x
    return all_imgs_list


def preprocess_data(which_model, batch_data_root, stage1_labels=None):
    # 不指定验证集时，x_val, y_val = None
    if not batch_data_root:
        x = None
        y = None
        return x, y
    if not os.path.exists(batch_data_root):
        print("请重新指定data_root!")
    if which_model == 1:
        # model_1
        if not stage1_labels:
            x, subject_id_set_list = preprocess_data_1(batch_data_root)
            x = preprocess_data_2(x)
            return x, subject_id_set_list
        x, y = preprocess_data_1(batch_data_root, stage1_labels)
        x, y = preprocess_data_2(x, y)
    elif which_model == 2:
        # model_2
        if not stage1_labels:
            x, subject_id_set_list = preprocess_data_1(batch_data_root)
            return x, subject_id_set_list
        x, y = preprocess_data_1(batch_data_root, stage1_labels)
    else:
        # multiple inputs
        if not stage1_labels:
            x, subject_id_set_list = preprocess_data_1(batch_data_root)
            x = preprocess_data_3(x)
            return x, subject_id_set_list
        x, y = preprocess_data_1(batch_data_root, stage1_labels)
        x = preprocess_data_3(x)
    return x, y
