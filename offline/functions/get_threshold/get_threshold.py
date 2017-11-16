#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : get_threshold.py
# @Author: zjj421
# @Date  : 17-10-15
# @Desc  :
import numpy as np
from pandas import Series, DataFrame

from offline.functions import sort_zone_id


def get_17_values_list(df):
    all_probability = df["Probability"]
    all_probability = np.asarray(all_probability)
    all_probability = all_probability.reshape(-1, 17)
    all_probability_list = np.array_split(all_probability, 17, axis=1)
    all_probability_list = list(map(lambda x: x.reshape(-1), all_probability_list))
    return all_probability_list


def get_sequence_01(x_list, threshold):
    sequece_01 = [1 if x >= threshold else 0 for x in x_list]
    return sequece_01


def get_threshold(predict_values, real_values, accuracy):
    predict_values = Series(predict_values)
    real_values = Series(real_values)
    predict_values = (predict_values * accuracy).apply(lambda x: int(x))
    errors = []
    for threshold in range(accuracy + 1):
        sequence_01 = get_sequence_01(predict_values, threshold)
        # sss = 0
        # for i in sequence_01:
        #     if i == 0:
        #         sss = sss + 1
        # print("sequence_01中0的个数为-------- {}".format(sss))
        error = sum(abs(sequence_01 - real_values))
        errors.append(error)
    m = min(errors)
    thresholds = []
    for i, v in enumerate(errors):
        if v == m:
            thresholds.append(i / accuracy)
    return min(thresholds), max(thresholds), m



def __main():
    df_submission = sort_zone_id(SUBMISSION_CSVFILE, save_ouput=False)
    df_stage1_labels = sort_zone_id(STAGE1_LABELS_CSVFILE, save_ouput=False)
    predict_values_17_list = get_17_values_list(df_submission)
    real_values_17_list = get_17_values_list(df_stage1_labels)
    thresholds = []
    for i in range(17):
        zone_x_predict_values = predict_values_17_list[i]
        zone_x_real_values = real_values_17_list[i]
        threshold_0, threshold_1, num_error = get_threshold(zone_x_predict_values, zone_x_real_values, ACCURACY)
        thresholds.append([threshold_0, threshold_1, num_error])

    # print("各区块的阈值为： ", thresholds)
    # zone_id = ["Zone{}".format(num) for num in range(1, 18)]
    df = DataFrame(thresholds, columns=["threshold_0", "threshold_1", "num_error"])
    df.to_csv("threshold.csv", index=False,
              sep=",")
    print(df)
    print("All have done.")


if __name__ == '__main__':
    SUBMISSION_CSVFILE = "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/predict2true/stage1_submission_model_1002_VGG19_200_0val_3_50_flag_train_data_predict_5pbs_prdt2t_0val_3_1000_flag2_predict_values_2_3pbs.csv"
    STAGE1_LABELS_CSVFILE = "/home/zj/helloworld/kaggle/tr/new_tr/functions/get_threshold/sorted_stage1_labels.csv"
    # STAGE1_LABELS_CSVFILE = "/home/zj/桌面/labels.csv"
    # SUBMISSION_CSVFILE = "/home/zj/桌面/predict_values.csv"
    ACCURACY = int(1E3)
    __main()
