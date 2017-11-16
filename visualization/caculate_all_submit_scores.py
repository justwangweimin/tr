#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : caculate_all_submit_scores.py
# @Author: zjj421
# @Date  : 17-11-15
# @Desc  :
import os
import re
import pandas as pd
from pandas import DataFrame

from offline.functions.caculate_scores import get_probability_values, my_binary_crossentropy


def get_submission_file(path):
    submission_file_path_list = []
    patt = re.compile(r'(^stage\d_submission.*)\.csv$')
    for root, dirs, files in os.walk(path):
        for file in files:
            result = patt.search(file)
            if result:
                file_path = os.path.join(root, file)
                submission_file_path_list.append(file_path)
    return submission_file_path_list


def __main():
    submission_file_path_list = get_submission_file(PATH)
    true_values = get_probability_values(TRUE_CSV)
    losses = []
    for file_path in submission_file_path_list:
        predict_values = get_probability_values(file_path)
        loss = my_binary_crossentropy(true_values, predict_values)
        losses.append(loss)
        # print("loss: ", loss)
        # print("loss: ", round(loss, 5))
    df = DataFrame(data={'file_path': submission_file_path_list, 'loss': losses})
    df = df.sort_values(by='loss')
    print(df)
    df.to_csv("score_list.csv", index=False)

    # print(submission_file_path_list)


if __name__ == '__main__':
    PATH = "/home/zj/helloworld/kaggle/tr/offline/results"
    TRUE_CSV = "/home/zj/helloworld/kaggle/tr/input/test_labels.csv"
    __main()
