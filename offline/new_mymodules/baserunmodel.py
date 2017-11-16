#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : baserunmodel.py
# @Author: zjj421
# @Date  : 17-11-9
# @Desc  :
import os
from datetime import datetime

from pandas import Series, DataFrame
import pandas as pd
from offline.new_mymodules.basedata import BaseData
from offline.new_mymodules.my_model import MyModel


class BaseRunModel(MyModel, BaseData):
    def __init__(self, which_model=None,
                 cnn_name=None,
                 img_height=200,
                 num_val=0,
                 batch_size=3,
                 epochs=12,
                 model_path=None,
                 predict_batch_size=3,
                 flag="1",
                 f_features_trained_path=None,
                 f_features_tested_path=None,
                 ):
        MyModel.__init__(self, which_model=which_model,
                         cnn_name=cnn_name,
                         img_height=img_height,
                         num_val=num_val,
                         batch_size=batch_size,
                         epochs=epochs,
                         model_path=model_path,
                         predict_batch_size=predict_batch_size,
                         flag=flag)
        BaseData.__init__(self, which_model=which_model,
                          cnn_name=cnn_name,
                          f_features_trained_path=f_features_trained_path,
                          f_features_tested_path=f_features_tested_path,
                          num_val=num_val)
        self.begin = datetime.now()
        print("开始时间： ", self.begin)

    def save_keys(self):
        if not os.path.exists(self.root):
            # makedirs可生成多级递归目录
            os.makedirs(self.root)
        csv_name = "train_val_ids_{}.csv".format(self.directory_name)
        csv_path = os.path.join(self.root, csv_name)
        # if os.path.exists(csv_path):
        #     print("请重新指定MyModel参数，{} 已存在！".format(csv_path))
        #     exit()
        # print(len(train_keys))
        # print(len(val_keys))
        train_keys = Series(self.train_keys)
        val_keys = Series(self.val_keys)
        data = {"train_ids": train_keys,
                "val_ids": val_keys}
        df = DataFrame(data)
        df.to_csv(csv_path,
                  index=False,
                  sep=",")

    def save_runtime(self):
        end = datetime.now()
        runtime = (end - self.begin).seconds
        print("结束时间： ", end)
        print("总耗时: {}s".format(runtime))
        output_fit_csv = os.path.join(self.root, self.output_fit_basename)
        df = pd.read_csv(output_fit_csv)
        df["runtime"] = runtime
        df.to_csv(output_fit_csv, index=False)
        print("运行时间已写进csv文件！")

    def train_model(self, *kwargs):
        super().train_model(*kwargs)
        self.save_runtime()
        self.save_keys()


def __main():
    pass


if __name__ == '__main__':
    __main()
