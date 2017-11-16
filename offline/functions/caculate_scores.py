#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : caculate_scores.py
# @Author: zjj421
# @Date  : 17-11-10
# @Desc  :
from math import log
import pandas as pd

from offline.functions.get_threshold.sort_zone_id import sort_zone_id


def my_binary_crossentropy(true_values, predict_values):
    sum = 0
    len_ = len(true_values)
    for true_value, predict_value in zip(true_values, predict_values):
        sum = sum + true_value * log(predict_value) + (1 - true_value) * log(1 - predict_value)
    loss = -sum / len_
    return loss


def get_probability_values(csv_path):
    df = sort_zone_id(csv_path)
    # df = pd.read_csv(csv_path)
    probability_values = list(df["Probability"])
    # print(probability_values)
    return probability_values


def __main():
    true_values = get_probability_values(TRUE_CSV)
    predict_values = get_probability_values(PREDICT_CSV)
    loss = my_binary_crossentropy(true_values, predict_values)
    # print("loss: ", loss)
    print("loss: ", round(loss, 5))


if __name__ == '__main__':
    PREDICT_CSV = "/home/zj/helloworld/kaggle/tr/offline/results/model_1005_VGG19_200_0val_3_12_flag_test/stage1_submission_model_1005_VGG19_200_0val_3_12_flag_test_3pbs.csv"
    TRUE_CSV = "/home/zj/helloworld/kaggle/tr/input/test_labels.csv"
    __main()
