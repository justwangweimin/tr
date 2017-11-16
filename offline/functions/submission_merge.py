#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : submission_merge.py
# @Author: zjj421
# @Date  : 17-10-27
# @Desc  :
import numpy as np
import pandas as pd


def submisson_merge(val_loss_list, submisson_file_list):
    val_loss_list = np.asarray(val_loss_list)
    # 置信度
    confidence_list = (1 - val_loss_list)
    confidence_list = confidence_list / sum(confidence_list)
    # df_probability = []
    _sum = 0
    df = None
    for i, v in enumerate(submisson_file_list):
        df = pd.read_csv(v)
        df["Probability"] = df["Probability"] * confidence_list[i]
        # df_probability.append(df["Probability"])
        _sum = _sum + df["Probability"]
    try:
        df["Probability"] = _sum
    except:
        del df["Probability"]
        df["Probability"] = _sum
        print("except语句运行了")
    df.to_csv("submission_merged.csv", index=False)


def __main():
    # val_loss_list = [0.246039, 0.247114, 0.293133, 0.314636]
    val_loss_list = [0.22944, 0.293133, 0.314636]
    submisson_file_list = [
        "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/stage1_submission_model_1002_VGG19_200_100val_3_50_flag_1_5pbs.csv",
        # "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/stage1_submission_model_1002_VGG16_200_100val_3_51_flag_1_5pbs.csv",
        "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/stage1_submission_model_1001_ResNet50_200_100val_3_12_flag_1_5pbs.csv",
        "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/stage1_submission_model_1001_InceptionV3_200_100val_3_120_flag_1_12pbs.csv"]
    submisson_merge(val_loss_list, submisson_file_list)


if __name__ == '__main__':
    __main()
