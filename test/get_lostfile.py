#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : get_lostfile.py
# @Author: zjj421
# @Date  : 17-9-11
# @Desc  :
from mymodule.data_random_classfication import collect_filename_from_dir, collect_subject_id_from_csv
import numpy as np
import os

def get_lostfile():
    file_list = collect_filename_from_dir("/media/zj/study/kaggle/stage1_aps")

    label_file_list = collect_subject_id_from_csv("/media/zj/study/kaggle/stage1_labels.csv")
    size_0_file_list = []
    for i in file_list:
        i = os.path.join("/media/zj/study/kaggle/stage1_aps", (i+".aps"))
        size = get_0_size_file(i)
        if size == 0:
            size_0_file_list.append(i)

    print(len(size_0_file_list))
    print(size_0_file_list)
    file_list = np.array(file_list)
    label_file_list = np.array(label_file_list)
    # 求交集
    file_uploaded = np.intersect1d(file_list, label_file_list)

    # 求集合的差
    file_lost = np.setdiff1d(label_file_list, file_uploaded)
    print("缺失文件个数：", len(file_lost))
    for i in file_lost:
        file = i + ".aps"
        print(file, end="\n")

    print("\nDone!")

def get_0_size_file(file_path):
    size = os.path.getsize(file_path)
    return size

get_lostfile()