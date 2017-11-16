#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : analysis_predict_values.py
# @Author: zjj421
# @Date  : 17-11-15
# @Desc  :
from offline.functions.caculate_scores import my_binary_crossentropy
from offline.functions.get_threshold.sort_zone_id import sort_zone_id


# 从csv文件中获取指定区块号的所有预测值,zone_nb取值为[1, 17]
def get_specified_zone_probability_values(csv_path, zone_nb):
    df = sort_zone_id(csv_path)
    df = df[df["Zone_id"] == zone_nb]
    probability_values = list(df["Probability"])
    return probability_values


def __main():
    loss_list = []
    for i in range(1, 18):
        true_values = get_specified_zone_probability_values(TRUE_CSV, i)
        predict_values = get_specified_zone_probability_values(PREDICT_CSV, i)
        loss = my_binary_crossentropy(true_values, predict_values)
        loss_list.append(round(loss, 5))
        # print("loss: ", loss)
        print("loss_{}: ".format(i), round(loss, 5))
    print("loss: ", round(sum(loss_list) / len(loss_list), 5))


if __name__ == '__main__':
    PREDICT_CSV = "/home/zj/helloworld/kaggle/tr/offline/results/model_3001_VGG19_200_0val_3_300_flag_tval/stage1_submission_model_3001_VGG19_200_0val_3_300_flag_tval_3pbs.csv"
    TRUE_CSV = "/home/zj/helloworld/kaggle/tr/input/test_labels.csv"
    # ZONE_NB = 3
    __main()
