#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : computescore.py
# @Author: wangweimin
# @Date  : 17-11-8
# @Desc  :
from math import log
import pandas as pd
import train.const as const
def binaryCrossentropy(true_values, predict_values):
    sum = 0
    len_ = len(true_values)
    precision = 10 ** 16
    zero = 1 / precision
    for true_value, predict_value in zip(true_values, predict_values):
        predict_value1 = predict_value
        if predict_value > 0.999:
            predict_value = 1 - zero
        elif predict_value < 0.001:
            predict_value = zero

        #sum1 = sum1 + true_value * log(predict_value1) + (1 - true_value) * log(1 - predict_value1)
        sum = sum + true_value * log(predict_value) + (1 - true_value) * log(1 - predict_value)
    loss = -sum / len_
    return int(precision * loss) / precision

def sortById(csvfile):
    df = pd.read_csv(csvfile)
    df["Subject_id"], df["Zone_id"] = df["Id"].str.split("_Zone", 1).str
    df["Zone_id"] = df["Zone_id"].apply(lambda x: int(x))
    df = df.sort_values(by=["Subject_id", "Zone_id"])
    return df

def getProbability(csvfile):
    df = sortById(csvfile)
    ret = list(df["Probability"])
    return ret


def run():
    true_values = getProbability(TESTLABELCSV)
    predict_values = getProbability(SUBMISSIONCSV)
    loss = binaryCrossentropy(true_values, predict_values)
    print("loss: ", loss)


SUBMISSIONCSV = const.SUBMISSIONCSV
TESTLABELCSV = const.TESTLABELCSV
if __name__ == '__main__':
    run()