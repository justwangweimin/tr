#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_random_classfication.py
# @Author: zjj421
# @Date  : 17-9-5
# @Desc  :
import os
import math
import pandas as pd
import numpy as np


# def split_header(csvfile_path, new_csvfile_path):
#     df = pd.read_csv(csvfile_path)
#     # column_index_list = list(df.columns)
#     # if column_index_list.count("Subject") == 0:
#     #     return csvfile_path
#
#     # Separate the zone and subject id into a df
#     print(df['Probability'])
#     df['Subject'], df['Zone'] = df['Id'].str.split('_', 1).str
#     df = df[['Subject', 'Zone', 'Probability']]
#     print(df['Probability'])
#     # print(df)
#     df.to_csv(new_csvfile_path)
#     return new_csvfile_path

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
# dir_path = "/media/zj/study/kaggle/stage1_aps"
# csvfile_path = "/media/zj/study/kaggle/stage1_sample_submission.csv"
# new_csvfile_path = "/media/zj/study/kaggle/new_stage1_sample_submission.csv"
# split_header(csvfile_path, new_csvfile_path)

def collect_filename_from_dir(dir_path):
    filename_list = []
    # root, dirs, files 分别表示：父目录名（全名），目录名，文件名
    # 如果dir_path不存在,循环内的语句不会被执行，而继续运行循环后面的语句
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            filename = os.path.splitext(file)[0]
            filename_list.append(filename)
    print("数据集中总共有{}个文件".format(len(filename_list)))
    # print(filename_list)
    return filename_list


def collect_subject_id_from_csv(csvfile_path):
    # 拆分表头，把id拆分成subject_id和zone_num
    df = pd.read_csv(csvfile_path)
    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_', 1).str
    df = df[['Subject', 'Zone', 'Probability']]
    # print(df)
    subject_id_list = list(df["Subject"])
    subject_id_list = list(set(subject_id_list))
    print("标签文件中有{}个不同的subject_id".format(len(subject_id_list)))
    # print(subject_id_set)
    return subject_id_list


def get_num_files_from_dirname(dirname):
    num_files = 0
    for root, dirs, files in os.walk(dirname):
        for file in files:
            num_files = num_files + 1
    return num_files


def create_symlink(src, dst):
    if not os.path.exists(src):
        print("源文件不存在，无法创建软链接！")
        return
    if os.path.isfile(src):
        dirname = os.path.dirname(dst)  # 返回dst的目录（父目录全名）
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    os.symlink(src, dst)

# 获取有标签的数据
def get_label_data_list(data_root, labels_path):
    filename_list = collect_filename_from_dir(data_root)
    subject_id_list = collect_subject_id_from_csv(labels_path)
    filename_list = np.array(filename_list)
    subject_id_list = np.array(subject_id_list)
    # 交集
    label_data_list = list(np.intersect1d(filename_list, subject_id_list))
    print("-" * 100)
    print("type(label_data_list):\n", type(label_data_list))
    print("len(label_data_list):\n", len(label_data_list))
    return label_data_list


def get_data_path_list(data_list_no_ext, data_root):
    file_ext = ".aps"
    data_root = data_root
    data_path_list = list(map(lambda f: os.path.join(data_root, (f + file_ext)), data_list_no_ext))
    print("length of data_path_list:\n", len(data_path_list))
    return data_path_list


def generate_new_data_root(data_path_list, new_data_root, files_per_dir):
    num_dirs = math.ceil(len(data_path_list) / files_per_dir)  # 上取整
    for i in range(num_dirs):
        if num_dirs != 1:
            new_batch_data_dir = "dir-{:0>3}".format(i + 1)
            new_batch_data_root = os.path.join(new_data_root, new_batch_data_dir)
            print("-" * 100)
            print("new_batch_data_root:\n", new_batch_data_root)
        else:
            new_batch_data_root = new_data_root
        for f in data_path_list[i * files_per_dir:(i + 1) * files_per_dir]:
            # print("==" * 100)
            # print("f=", f)
            new_train_data_path = os.path.join(new_batch_data_root, os.path.split(f)[1])
            create_symlink(f, new_train_data_path)


def generate_new_train_data_root(label_data_list, data_root, syblink_train_data_root,
                                 num_files_per_dir_train):
    num_files = get_num_files_from_dirname(syblink_train_data_root)
    if num_files != 0:
        print("链接目录中已有文件，请重新指定链接数据目录！")
        return
    # num_data_set = len(list(label_data_list))
    # num_train_data = round(num_data_set * 0.8)
    # train_data_list = label_data_list[:num_train_data]
    train_data_list = label_data_list[100:]
    print("其中训练集的长度为:{}".format(len(train_data_list)))
    train_data_path_list = get_data_path_list(train_data_list, data_root)
    generate_new_data_root(train_data_path_list, syblink_train_data_root, num_files_per_dir_train)


def generate_new_validation_data_root(label_data_list, data_root, syblink_validation_data_root):
    num_files = get_num_files_from_dirname(syblink_validation_data_root)
    if num_files != 0:
        print("链接目录中已有文件，请重新指定链接数据目录！")
        return
    # num_data_set = len(list(label_data_list))
    # num_train_data = round(num_data_set * 0.8)
    # validation_data_list = label_data_list[num_train_data:]
    validation_data_list = label_data_list[:100]
    num_validation_data = len(validation_data_list)
    print("其中验证集的长度为:{}".format(num_validation_data))
    validation_data_path_list = get_data_path_list(validation_data_list, data_root)
    generate_new_data_root(validation_data_path_list, syblink_validation_data_root, num_validation_data)


def _main():
    label_data_list = get_label_data_list(DATA_ROOT, LABELS_PATH)
    # print("打乱前：\n", label_data_list)
    # 就地打乱数据集
    np.random.shuffle(label_data_list)
    # print("打乱后：\n", label_data_list)
    num_data_set = len(list(label_data_list))
    print("数据集的长度为:{}".format(num_data_set))

    # 生成训练集
    generate_new_train_data_root(label_data_list, DATA_ROOT, SYBLINK_TRAIN_DATA_ROOT,
                                 NUM_FILES_PER_DIR_TRAIN)
    # 生成验证集
    generate_new_validation_data_root(label_data_list, DATA_ROOT, SYBLINK_VALIDATION_DATA_ROOT)
    print("All have done.")


if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------------------------
    # 以下内容需要手动指定
    NUM_FILES_PER_DIR_TRAIN = 1147
    WHICH_MODEL = 5
    NTH_WHICH_MODEL = 6
    NUM_VAL = 100
    BATCH_SIZE = 4
    EPOCHS = 12
    file = "model{which_model}_{nth_which_model}_{num_val}val_{batch_size}_{epochs}".format(
        which_model=WHICH_MODEL,
        nth_which_model=NTH_WHICH_MODEL,
        num_val=NUM_VAL,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    SYBLINK_TRAIN_DATA_ROOT = "/home/zj/helloworld/kaggle/threat_recognition/syblink_stage1_aps/{}/train_data_root".format(
        file)
    SYBLINK_VALIDATION_DATA_ROOT = "/home/zj/helloworld/kaggle/threat_recognition/syblink_stage1_aps/{}/validation_data_root".format(
        file)
    # -------------------------------------------------------------------------------------------------------------------
    DATA_ROOT = "/media/zj/study/kaggle/stage1_aps"
    LABELS_PATH = "/media/zj/study/kaggle/stage1_labels.csv"
    _main()
