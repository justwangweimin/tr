#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_model.py
# @Author: zjj421
# @Date  : 17-10-13
# @Desc  :
from datetime import datetime
from keras import Input, models
from keras.layers import Dense, concatenate
import h5py
import numpy as np

import os
import pandas as pd


MODEL_SAVED_DIRNAME = "."
OUTPUT_FIT_DIANAME = "."


class MyModel:
    def __init__(self, which_model, cnn_name, img_height=200, num_val=0, batch_size=3, epochs=12, no=1,
                 model_path=None, output_fit_dirname=OUTPUT_FIT_DIANAME, model_saved_dirname=MODEL_SAVED_DIRNAME,
                 predict_batch_size=None):
        self.which_model = which_model
        self.cnn_name = cnn_name
        self.img_height = img_height
        self.num_val = num_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.no = no
        self.model_path = model_path
        self.output_fit_dirname = output_fit_dirname
        self.model_saved_dirname = model_saved_dirname
        self.predict_batch_size = predict_batch_size

    @property
    def model_name(self):
        model_name = "model_{which_model}_{cnn_name}_{img_height}_{num_val}val_{batch_size}_{epochs}_no_{no}.h5".format(
            which_model=self.which_model,
            cnn_name=self.cnn_name,
            img_height=self.img_height,
            num_val=self.num_val,
            batch_size=self.batch_size,
            epochs=self.epochs,
            no=self.no)
        return model_name

    @property
    def output_fit_basename(self):
        output_fit = "output_fit_{}.csv".format(self.model_name.split(".")[0])
        return output_fit

    @property
    def stage1_submission_basename(self):
        stage1_submission = "stage1_submission_{}_{}pbs.csv".format(self.model_name.split(".")[0],
                                                                    self.predict_batch_size)
        return stage1_submission

    def create_model(self):
        model = create_model(self.which_model)
        if self.model_path:
            if os.path.exists(self.model_path):
                model.load_weights(self.model_path)
                print("模型权重'{}'导入成功".format(self.model_path))
            else:
                print("请重新指定model_path!")
                exit()
        return model

    def train_model(self, model, x_train, y_train, x_val=None, y_val=None, validation_split=0.):
        model = model
        if x_val and y_val:
            hist = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, shuffle=True,
                             validation_data=(x_val, y_val))
        elif validation_split > 0.:
            hist = model.fit(x_train, y_train, validation_split=validation_split, batch_size=self.batch_size,
                             epochs=self.epochs, shuffle=True)
        else:
            hist = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, shuffle=True)
        output_fit_csv = os.path.join(self.output_fit_dirname, self.output_fit_basename)
        self._save_output_fit(hist.history, output_fit_csv)
        model_save_path = os.path.join(self.model_saved_dirname, self.model_name)
        model.save_weights(model_save_path)
        print("模型已经保存。")
    
    def predict_model(self, model, x_test, ids):
        predict_values = model.predict(x_test, batch_size=self.predict_batch_size, verbose=1)
        predict_values = predict_values.reshape(-1)
        dataframe = pd.DataFrame({"Id":ids, "Probability":predict_values})
        stage1_submission_csv = os.path.join(".", self.stage1_submission_basename)
        dataframe.to_csv(
            stage1_submission_csv,
            index=False,sep=",")
        

    def _save_output_fit(self, hist, output_fit_csv):
        df = pd.DataFrame()
        if "loss" in hist:
            df["loss"] = hist["loss"]
        if "val_loss" in hist:
            df["val_loss"] = hist["val_loss"]
        df.to_csv(output_fit_csv, index=False)
        print("*" * 100)
        print('输出结果 "{}" 已保存！'.format(output_fit_csv))

def create_model_1001():
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(2048,))
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(2048, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# 新模型，特征融合
def create_model_2001():
    # Input层
    inputs = []
    num_angles = 16
    num_cnns = 2
    for i in range(num_cnns):
        for j in range(num_angles):
            input = Input(shape=(2048,))
            inputs.append(input)
    frame_features_1 = concatenate(inputs[0:num_angles])
    frame_features_1 = Dense(2048, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_2 = concatenate(inputs[num_angles:num_angles * 2])
    frame_features_2 = Dense(2048, activation="tanh")(frame_features_2)
    frame_features_2 = Dense(256, activation="tanh")(frame_features_2)
    frame_features = concatenate([frame_features_1, frame_features_2])
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model
# 新模型，特征融合
def create_model_2002():
    # Input层
    inputs = []
    num_angles = 16
    num_cnns = 3
    for i in range(num_cnns):
        for j in range(num_angles):
            input = Input(shape=(2048,))
            inputs.append(input)
    frame_features_1 = concatenate(inputs[0:num_angles])
    frame_features_1 = Dense(2048, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_2 = concatenate(inputs[num_angles:num_angles * 2])
    frame_features_2 = Dense(2048, activation="tanh")(frame_features_2)
    frame_features_2 = Dense(256, activation="tanh")(frame_features_2)
    frame_features_3 = concatenate(inputs[num_angles*2:num_angles * 3])
    frame_features_3 = Dense(2048, activation="tanh")(frame_features_3)
    frame_features_3 = Dense(256, activation="tanh")(frame_features_3)
    frame_features = concatenate([frame_features_1, frame_features_2, frame_features_3])
    frame_features = Dense(256, activation="tanh")(frame_features)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


def create_model(which_model):
    if which_model == 1001:
        model = create_model_1001()
    elif which_model == 2001:
        model = create_model_2001()
    elif which_model == 2002:
        model = create_model_2002()
    else:
        model = None
        print("请设置好WHICH_MODEL!")
        exit()
    # 打印模型
    model.summary()
    print("模型创建完毕！")
    print("-" * 100)
    return model


def prepare_data(f_features, f_labels, cnn_name):
    f_features = f_features[cnn_name+"_features"]
    cnn_features = []
    labels = []
    for i, key in enumerate(f_features.keys()):
        cnn_feature = f_features[key].value
        cnn_features.append(cnn_feature)
        label = f_labels[key].value
        labels.append(label)
    cnn_features = np.asarray(cnn_features, dtype=np.float32)
    cnn_features_list = np.array_split(cnn_features, 16, axis=1)
    dim_1 = cnn_features.shape[-1]
    for j, x in enumerate(cnn_features_list):
        x = x.reshape(-1, dim_1)
        cnn_features_list[j] = x
    labels = np.asarray(labels)
    return cnn_features_list, labels

def prepare_test_data(f_features, cnn_name):
    f_features = f_features[cnn_name+"_features"]
    cnn_features = []
    ids = []
    for i, key in enumerate(f_features.keys()):
        cnn_feature = f_features[key].value
        cnn_features.append(cnn_feature)
        for n in range(1, 18):
            id = "{}_Zone{}".format(key, n)
            ids.append(id)
    cnn_features = np.asarray(cnn_features, dtype=np.float32)
    cnn_features_list = np.array_split(cnn_features, 16, axis=1)
    dim_1 = cnn_features.shape[-1]
    for j, x in enumerate(cnn_features_list):
        x = x.reshape(-1, dim_1)
        cnn_features_list[j] = x
    return cnn_features_list, ids
def train_model():
    mymodel_1 = MyModel(which_model=2001,
                      cnn_name="ResNet50",
                      img_height=200,
                      num_val=100,
                      batch_size=3,
                      epochs=12,
                      no=11110111,
                      model_path=r".\model_2001_ResNet50_200_100val_3_120_no_11110111.h5",
                      # output_fit_dirname=OUTPUT_FIT_DIANAME,
                      # model_saved_dirname=MODEL_SAVED_DIRNAME,
                      predict_batch_size=5)
    mymodel_2 = MyModel(which_model=2001,
                      cnn_name="Xception",
                      img_height=200,
                      num_val=100,
                      batch_size=3,
                      epochs=12,
                      no=1,
                      model_path=None,
                      # output_fit_dirname=OUTPUT_FIT_DIANAME,
                      # model_saved_dirname=MODEL_SAVED_DIRNAME,
                      predict_batch_size=None)
#     mymodel_3 = MyModel(which_model=2002,
#                       cnn_name="InceptionV3",
#                       img_height=200,
#                       num_val=100,
#                       batch_size=3,
#                       epochs=120,
#                       no=1,
#                       model_path=None,
#                       # output_fit_dirname=OUTPUT_FIT_DIANAME,
#                       # model_saved_dirname=MODEL_SAVED_DIRNAME,
#                       predict_batch_size=None)
    model = mymodel_1.create_model()
    cnn_name_1 = mymodel_1.cnn_name
#     x_train_1, y_train_1 = prepare_data(F_FEATURES, F_LABELS, cnn_name_1)
    cnn_name_2 = mymodel_2.cnn_name
#     x_train_2, y_train_2 = prepare_data(F_FEATURES, F_LABELS, cnn_name_2)
# #     cnn_name_3 = mymodel_3.cnn_name
# #     x_train_3, y_train_3 = prepare_data(F_FEATURES, F_LABELS, cnn_name_3)
#     x_train_1.extend(x_train_2)  
#     x_train_1.extend(x_train_3)  
#     y_train_1 = list(y_train_1)
#     y_train_1.extend(y_train_2)
#     y_train_1 = np.array(y_train_1)
#     mymodel_1.train_model(model, x_train_1, y_train_1, validation_split=0.087)
    
    x_test_1, ids = prepare_test_data(F_FEATURES_TESTED, cnn_name_1)
    x_test_2, ids = prepare_test_data(F_FEATURES_TESTED, cnn_name_2)
    x_test_1.extend(x_test_2)
    mymodel_1.predict_model(model, x_test_1, ids)
def __main():
    train_model()
    
    


if __name__ == '__main__':
    F_IMGS = h5py.File(r".\imgs_preprocessed_200_200_3.h5", "r")
    F_LABELS = F_IMGS["labels"]
    F_FEATURES = h5py.File(r".\cnn_features.h5", "r")
    F_FEATURES_TESTED = h5py.File(r".\cnn_features_200_tested.h5", "r")
    begin = datetime.now()
    print("开始时间： ", begin)
    __main()
    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))
