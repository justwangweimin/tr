#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : plot_from_csv.py
# @Author: zjj421
# @Date  : 17-9-22
# @Desc  :
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_figure(csvfile_1):
    df = pd.read_csv(csvfile_1)
    x = np.linspace(0, 6000, 6000)
    loss = df["loss"]
    val_loss = df["val_loss"]
    title = os.path.splitext(os.path.basename(csvfile_1))[0]
    plt.title(title)
    plt.grid()
    plt.plot(x, loss, "r", label='loss')
    plt.plot(x, val_loss, "g", label='val_loss')
    # plt.show()


def plot_figure_2(csvfile_1):
    df = pd.read_csv(csvfile_1)
    loss = df["loss"]
    val_loss = df["val_loss"]
    title = os.path.splitext(os.path.basename(csvfile_1))[0]
    plt.title(title)
    plt.grid()
    len_ = len(loss)
    print(len_)
    x = np.linspace(0, len_, len_)
    plt.plot(x, loss, "r--", label='loss')
    plt.plot(x, val_loss, "g--", label='val_loss')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # CSV = "/home/zj/helloworld/kaggle/tr/new_tr/output_fit/output_fit_model_1002_VGG19_200_100val_3_50_flag_1.csv"
    # CSV_1 = "/home/zj/helloworld/kaggle/tr/new_tr/output_fit/output_fit_model_1002_VGG19_200_100val_3_120_flag_1.csv"
    # CSV_1 = "/home/zj/helloworld/kaggle/tr/new_tr/output_fit/output_fit_model_1005_VGG19_200_100val_1047_6000_flag_1.csv"
    CSV_1 = "/home/zj/helloworld/kaggle/tr/new_tr/output_fit/output_fit_model_1002_VGG19_200_100val_1047_6000_flag_1.csv"
    # CSV_2 = "/home/zj/helloworld/kaggle/tr/new_tr/output_fit/output_fit_model_1005_VGG19_200_100val_1047_120000_flag_1.csv"
    CSV_2 = "/home/zj/helloworld/kaggle/tr/new_tr/output_fit/output_fit_predict2true/output_fit_model_1002_VGG19_200_0val_3_50_flag_train_data_predict_5pbs_prdt2t_917_61133_flag2_1.csv"
    # plot_figure(CSV_1)
    plot_figure_2(CSV_2)


