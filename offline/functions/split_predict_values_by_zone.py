#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : split_predict_values_by_zone.py
# @Author: zjj421
# @Date  : 17-10-16
# @Desc  :
import pandas as pd
from pandas import DataFrame, Series


def __main():
    df = pd.read_csv(CSVFILE)
    df["Id_1"], df["Id_2"] = df["Id"].str.split("_Zone", 1).str
    # print(df)
    df_1 = df.ix[df["Id_2"] == str(1), ["Id", "Probability"]]
    df_1.to_csv("./zone1_test.csv",
              index=False, sep=",")
    # print(df_1)
    # for i in df_1:
    #     print(i)

if __name__ == '__main__':
    CSVFILE = "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/stage1_submission_model_1002_VGG19_200_0val_3_50_flag_train_data_predict_5pbs.csv"
    __main()
