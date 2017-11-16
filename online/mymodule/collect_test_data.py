#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : collect_test_data.py
# @Author: zjj421
# @Date  : 17-9-15
# @Desc  :
import os

import numpy as np

from online.mymodule.data_random_classfication import collect_filename_from_dir, collect_subject_id_from_csv, \
    generate_new_data_root


def collect_test_data(data_root, labels_path, file_ext=".aps"):
    filename_list = collect_filename_from_dir(data_root)
    subject_id_list = collect_subject_id_from_csv(labels_path)
    filename_list = np.array(filename_list)
    subject_id_list = np.array(subject_id_list)
    # 集合的差
    test_data_list = list(np.setdiff1d(filename_list, subject_id_list))
    data_path_list = list(map(lambda f: os.path.join(data_root, (f + file_ext)), test_data_list))
    print("共有{}个测试集".format(len(data_path_list)))
    return data_path_list


def __main():
    data_path_list = collect_test_data(DATA_ROOT, LABELS_PATH)
    generate_new_data_root(data_path_list, NEW_DATA_ROOT, NUM_TEST_DATA)


if __name__ == '__main__':
    DATA_ROOT = "/media/zj/study/kaggle/stage1_aps"
    LABELS_PATH = "/media/zj/study/kaggle/stage1_labels.csv"
    NEW_DATA_ROOT = "/home/zj/helloworld/kaggle/threat_recognition/syblink_stage1_aps/test_data_root"
    NUM_TEST_DATA = 100
    __main()
