#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : panda_read_csv.py
# @Author: zjj421
# @Date  : 17-9-1
# @Desc  :

import pandas as pd
import numpy as np


def get_subject_labels(infile, subject_id):
    # read labels into a dataframe
    df = pd.read_csv(infile)
    # print(np.arange(5, 10))
    # df.index = np.arange(10, df.shape[0] + 10)
    # s1 = df.loc[10, "Probability"]
    # s2 = df.loc[10:15, "Probability"]
    # s3 = df.loc[[10, 15], "Probability"]
    # print("-" * 20)
    # print(s1)
    # print("-" * 20)
    # print(s2)
    # print("-" * 20)
    # print(s3)
    # print("-" * 20)
    # print(df.shape)

    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_', 1).str
    df = df[['Subject', 'Zone', 'Probability']]
    # threat_list = df.loc[df['Zone'] == subject_id]
    threat_list = df.loc[df['Subject'] == subject_id]
    _subject_id = df['Subject']
    _subject_id = list(_subject_id)
    print(type(_subject_id))
    print(_subject_id)
    print("_"*40)
    print(list(df.columns).count('Subject'))
    # print("\n"*3)
    # s = df.loc[1, "Zone"]
    # print(s)

    return threat_list


def collect_filename_from_files(dir_path):
    filename_list = []
    # root, dirs, files 分别表示：父目录名（全名），目录名，文件名
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            filename = os.path.splitext(file)[0]
            filename_list.append(filename)
    print(filename_list)
    return filename_list


def split_header(csvfile_path, new_csvfile_path):
    df = pd.read_csv(csvfile_path)
    # column_index_list = list(df.columns)
    # if column_index_list.count("Subject") == 0:
    #     return csvfile_path

    # Separate the zone and subject id into a df
    print(df['Probability'])
    df['Subject'], df['Zone'] = df['Id'].str.split('_', 1).str
    df = df[['Subject', 'Zone', 'Probability']]
    print(df['Probability'])
    # print(df)
    df.to_csv(new_csvfile_path)
    return new_csvfile_path

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
# dir_path = "/media/zj/study/kaggle/stage1_aps"
# csvfile_path = "/media/zj/study/kaggle/stage1_sample_submission.csv"
# new_csvfile_path = "/media/zj/study/kaggle/new_stage1_sample_submission.csv"
# split_header(csvfile_path, new_csvfile_path)


def collect_subject_id_from_csv(csvfile_path, new_csvfile_path):
    # 拆分表头，把id拆分成subject_id和zone_num
    csvfile_path = split_header(csvfile_path, new_csvfile_path)
    df = pd.read_csv(csvfile_path)
    # print(df)
    subject_id_list = list(df["Subject"])
    print("length of subject_id_list:{}".format(len(subject_id_list)))
    return subject_id_list

# 找出没有标签的验证集
def data_set_classifiction(dir_path, csvfile_path, new_csvfile_path):
    filename_list = collect_filename_from_files(dir_path)
    subject_id_list = collect_subject_id_from_csv(csvfile_path)
    no_label_data_set = []
    for filename in filename_list:
        if subject_id_list.count(filename) == 0:
            no_label_data_set.append(filename)
    print("number of no_label_data_set:{}".format(len(no_label_data_set)))


def read_csv(infile):
    df = pd.read_csv(infile)
    return df

if __name__ == '__main__':
    infile = "/home/zj/helloworld/kaggle/threat_recognition/input/stage1_labels.csv"
    subject_id = "0043db5e8c819bffc15261b1f1ac5e42"
    new_csvfile = "/home/zj/testforfun/csvfile.csv"
    threat_list = get_subject_labels(infile, subject_id)
    # file1 = threat_list['Subject'].to_csv(new_csvfile)
    # print(threat_list)
    # print("-"*40)
    # df = read_csv(new_csvfile)
    # print(df)

