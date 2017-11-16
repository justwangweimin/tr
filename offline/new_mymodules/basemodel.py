#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : basemodel.py
# @Author: zjj421
# @Date  : 17-11-4
# @Desc  :



import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import save_model, load_model

from online.mymodule.create_model import create_model


class BaseModel:
    def __init__(self, which_model=None,
                 num_val=0,
                 batch_size=3,
                 epochs=12,
                 model_path=None,
                 predict_batch_size=3,
                 flag="1", ):
        self.which_model = which_model
        self.num_val = num_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        self.predict_batch_size = predict_batch_size
        self.flag = flag
        if not self.model_path:
            # 保证模型文件不被意外覆盖
            self._is_modelfile_repeat()
        self.model = self.generate_model()

    # 只读属性
    @property
    def directory_name(self):
        directory_name = ""
        return directory_name

    @property
    def model_name(self):
        model_name = "{}.h5".format(self.directory_name)
        return model_name

    @property
    def output_fit_basename(self):
        output_fit = "output_fit_{}.csv".format(self.directory_name)
        return output_fit

    @property
    def plot_output_fit_basename(self):
        output_fit = "plot_output_fit_{}.png".format(self.directory_name)
        return output_fit

    @property
    def submission_basename(self):
        if not self.predict_batch_size:
            print("请指定predice_batch_size!!")
            exit()
        submission_basename = "stage1_submission_{}_{}pbs.csv".format(self.directory_name,
                                                                      self.predict_batch_size)
        return submission_basename

    @property
    def root(self):
        dirname = os.path.dirname(os.getcwd())
        dirname = os.path.join(dirname, "results")
        dirname = os.path.join(dirname, self.directory_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return dirname

    def generate_model(self):
        if not self.model_path:
            model = create_model(self.which_model)
        else:
            if os.path.exists(self.model_path):
                model = load_model(self.model_path)
                print("模型权重'{}'导入成功".format(self.model_path))
            else:
                model = None
                print("请重新指定model_path!")
                exit()
        return model

    def train_model(self, x_train, y_train, x_val=None, y_val=None):
        if x_val == None and y_val == None:
            validation_data = None
        else:
            validation_data = (x_val, y_val)
        validation_split = self.num_val / len(x_train)
        hist = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, shuffle=True,
                              validation_data=validation_data, validation_split=validation_split)
        model_save_path = os.path.join(self.root, self.model_name)
        save_model(self.model, model_save_path, overwrite=True, include_optimizer=True)
        # model.save_weights(model_save_path)
        output_fit_csv = os.path.join(self.root, self.output_fit_basename)
        self._save_output_fit(hist.history, output_fit_csv)
        self.plot_output_fit(output_fit_csv)
        print('模型 "{}" 已经保存。'.format(model_save_path))

    def evaluate_model(self, x_val, y_val):
        val_loss = self.model.evaluate(x_val, y_val, self.batch_size)
        print("\nval_loss: {}".format(int(10 ** 5 * val_loss) / (10 ** 5)))

    def predict_model(self, x_test, ids, save_output=True):
        predict_values = self.model.predict(x_test, batch_size=self.predict_batch_size, verbose=1)
        print("\nmodel.predict的输出shape: ", predict_values.shape)
        if save_output:
            predict_values = predict_values.reshape(-1)
            dataframe = pd.DataFrame({"Id": ids, "Probability": predict_values})

            stage1_submission_csv = os.path.join(self.root, self.submission_basename)
            dataframe.to_csv(
                stage1_submission_csv,
                index=False, sep=",")
            print('文件"{}"已保存！'.format(stage1_submission_csv))
            print(predict_values.shape)

    def _save_output_fit(self, hist, output_fit_csv):
        df = pd.DataFrame()
        if "loss" in hist:
            df["loss"] = hist["loss"]
        if "val_loss" in hist:
            df["val_loss"] = hist["val_loss"]
        df.to_csv(output_fit_csv, index=False)
        print("*" * 100)
        print('输出结果 "{}" 已保存！'.format(output_fit_csv))

    def _get_filename_list(self):
        filename_list = []
        for root, dirs, files in os.walk(os.path.dirname(self.root)):
            for file in files:
                filename_list.append(file)
        return filename_list

    def _is_modelfile_repeat(self):
        filename_list = self._get_filename_list()
        print("filename_list: ", filename_list)
        print(self.model_name)
        if self.model_name in filename_list:
            print("模型名已存在，请检查模型参数设置")
            exit()

    # def _get_filename_list(self):
    #     file_name_list = []
    #     for root, dirs, files in os.walk(self.root):
    #         for file in files:
    #             file_name_list.append(file)
    #     return file_name_list

    # def _is_modelfile_repeat(self):
    #     filename_list = self._get_filename_list()
    #     if self.model_name in filename_list:
    #         print("模型名已存在，请检查模型参数设置")
    #         exit()

    def plot_output_fit(self, csv_path):
        df = pd.read_csv(csv_path)
        if "loss" in df.columns:
            loss = df["loss"]
        else:
            loss = None
        if "val_loss" in df.columns:
            val_loss = df["val_loss"]
        else:
            val_loss = None
        title = os.path.splitext(os.path.basename(csv_path))[0]
        plt.title(title)
        plt.grid()
        len_ = len(loss)
        plt.xlabel("epochs")
        plt.ylabel("error")
        x = np.linspace(0, len_, len_)
        if "loss" in df.columns:
            plt.plot(x, loss, "r", label='loss')
        if "val_loss" in df.columns:
            plt.plot(x, val_loss, "g", label='val_loss')
        plt.legend()
        savefig_path = os.path.join(self.root, self.plot_output_fit_basename)
        plt.savefig(savefig_path)
        # 可选择是否显示图片。
        plt.show()
