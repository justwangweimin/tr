#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : save_outputs.py
# @Author: zjj421
# @Date  : 17-9-23
# @Desc  :
import pandas as pd
import os


def save_output_fit(hist, current_batch_data_root, output_fit_csv):
    batch_data_num = os.path.splitext(os.path.basename(current_batch_data_root))[0]
    if os.path.exists(output_fit_csv):
        df = pd.read_csv(output_fit_csv)
    else:
        df = pd.DataFrame()
    loss = "loss_{}".format(batch_data_num)
    val_loss = "val_loss_{}".format(batch_data_num)
    if "loss" in hist:
        df[loss] = hist["loss"]
    if "val_loss" in hist:
        df[val_loss] = hist["val_loss"]
    df.to_csv(output_fit_csv, index=False)
    print("*" * 100)
    print('输出结果 "{}" 已保存！'.format(output_fit_csv))
