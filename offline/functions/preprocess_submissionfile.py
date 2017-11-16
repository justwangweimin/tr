#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : preprocess_submissionfile.py
# @Author: zjj421
# @Date  : 17-10-17
# @Desc  :
import pandas as pd

def get_p_n(x):
    if x > 0:
        x = 0.999999
    else:
        x = 0.000001
    return x


def __main():
    df_threshold = pd.read_csv(THRESHOLD_CSV)["threshold_0"]
    df_submission = pd.read_csv(SUBMISSION_CSV)
    df_1 = (df_submission - df_threshold).apply(get_p_n)
    print(df_1)

if __name__ == '__main__':
    THRESHOLD_CSV = ""
    SUBMISSION_CSV = ""
    __main()