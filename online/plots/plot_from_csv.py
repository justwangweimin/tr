#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : plot_from_csv.py
# @Author: zjj421
# @Date  : 17-9-22
# @Desc  :
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_figure(csvfile):
    df = pd.read_csv(csvfile)
    loss_names = [df["loss_dir-001"], df["loss_dir-002"], df["loss_dir-003"], df["loss_dir-004"], df["loss_dir-005"]]
    val_loss_names = [df["val_loss_dir-001"], df["val_loss_dir-002"], df["val_loss_dir-003"], df["val_loss_dir-004"],
                      df["val_loss_dir-005"]]
    x = np.linspace(0, 12, 12)
    title = os.path.splitext(os.path.basename(csvfile))[0]
    plt.title(title)
    plt.grid()
    loss_style = ["r", "g", "b", "k", "y"]
    val_loss_styles = ["r--", "g--", "b--", "k--", "y--"]
    for i in range(5):
        plt.plot(x, loss_names[i], loss_style[i])
        plt.plot(x, val_loss_names[i], val_loss_styles[i])
    plt.show()


if __name__ == '__main__':
    # CSV = "/home/zj/helloworld/kaggle/tr/output_fit/output_fit_model5_5_100val_6_12.csv"
    CSV = "/home/zj/helloworld/kaggle/tr/output_fit/output_fit_model2_5_100val_3_12.csv"
    plot_figure(CSV)


