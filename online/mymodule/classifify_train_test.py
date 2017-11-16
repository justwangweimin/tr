#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : classifify_train_test.py
# @Author: zjj421
# @Date  : 17-10-8
# @Desc  :
import numpy as np

from online.mymodule.data_random_classfication import get_label_data_list, get_num_files_from_dirname, \
    get_data_path_list, \
    generate_new_data_root, collect_filename_from_dir, collect_subject_id_from_csv


def classify_data(data_list, syblink_data_root):
    # 防止误操作，覆盖数据
    num_files = get_num_files_from_dirname(syblink_data_root)
    if num_files != 0:
        print("链接目录中已有文件，请重新指定链接数据目录！")
        return
    num_train_data = len(data_list)
    generate_new_data_root(data_list, syblink_data_root, num_train_data)


def get_test_data_list(data_root, labels_path):
    filename_list = collect_filename_from_dir(data_root)
    subject_id_list = collect_subject_id_from_csv(labels_path)
    filename_list = np.array(filename_list)
    subject_id_list = np.array(subject_id_list)
    # 求集合的差
    test_data_list = list(np.setdiff1d(filename_list, subject_id_list))
    return test_data_list


def _main():
    # 划分出训练集
    label_data_list = get_label_data_list(DATA_ROOT, STAGE1_LABELS)
    train_data_path_list = get_data_path_list(label_data_list, DATA_ROOT)
    print("数据集的长度为:{}".format(len(label_data_list)))
    classify_data(train_data_path_list, SYBLINK_TRAIN_DATA_ROOT)
    # 划分出测试集
    test_data_list = get_test_data_list(DATA_ROOT, STAGE1_LABELS)
    test_data_path_list = get_data_path_list(test_data_list, DATA_ROOT)
    classify_data(test_data_path_list, SYBLINK_TEST_DATA_ROOT)
    print("All have done.")


if __name__ == '__main__':
    DATA_ROOT = "/media/zj/study/kaggle/stage1_aps"
    STAGE1_LABELS = "/media/zj/study/kaggle/stage1_labels.csv"
    SYBLINK_TRAIN_DATA_ROOT = "/media/zj/study/kaggle/syblink_stage1_aps/syblink_train_data_root"
    SYBLINK_TEST_DATA_ROOT = "/media/zj/study/kaggle/syblink_stage1_aps/syblink_test_data_root"
    # _main()
    num_files_1 = get_num_files_from_dirname(SYBLINK_TRAIN_DATA_ROOT)
    num_files_2 = get_num_files_from_dirname(SYBLINK_TEST_DATA_ROOT)
    print(num_files_1, num_files_2)