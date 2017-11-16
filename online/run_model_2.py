#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_model_2.py
# @Author: zjj421
# @Date  : 17-10-12
# @Desc  :
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from offline.new_mymodules import Name
from online.mymodule.create_model import create_model


def train_model(model, batch_size, epochs, model_save_path, output_fit_csv):
    num_samples_per_batch = 1147
    features = []
    labels = []
    flag = 0
    len_subjects = len(F_XCEPTION_FEATURES.keys())
    for i, key in enumerate(F_XCEPTION_FEATURES.keys(), start=1):
        # vgg16_feature = F_VGG16_FEATURES[key].value
        resnet50_feature = F_XCEPTION_FEATURES[key].value
        # inceptionv3_feature = F_INCEPTIONV3_FEATURES[key].value
        # cnn_feature = np.concatenate((resnet50_feature, inceptionv3_feature), axis=0)
        print(i)
        # features.append(cnn_feature)
        features.append(resnet50_feature)
        label = F_LABELS[key].value
        labels.append(label)
        if i % num_samples_per_batch == 0 or i == len_subjects:
            features = np.asarray(features, dtype=np.float32)
            features_list = np.array_split(features, 16, axis=1)
            dim_1 = features.shape[-1]
            for j, x in enumerate(features_list):
                x = x.reshape(-1, dim_1)
                features_list[j] = x
            labels = np.asarray(labels)
            # print(labels.shape)
            # features_list = np.asarray(features_list, dtype=np.float32)
            # print(features_list.shape)
            hist = model.fit(features_list, labels, batch_size=batch_size, epochs=epochs, shuffle=True)
            save_output_fit(hist.history, output_fit_csv)
            model.save_weights(model_save_path)
            print("模型已经保存。")
            features = []
            labels = []
            flag = flag + 1
            # if flag == 3:
            #     exit()


def save_output_fit(hist, output_fit_csv):
    df = pd.DataFrame()
    df["loss"] = hist["loss"]
    df.to_csv(output_fit_csv, index=False)
    print("*" * 100)
    print('输出结果 "{}" 已保存！'.format(output_fit_csv))


def __main():
    name = Name(which_model=1001, cnn_name="xce", img_height=200, num_val=0, batch_size=3, epochs=120, no=1)
    model_name = name.model_name
    model_saved_path = os.path.join(MODEL_SAVE_PATH_DIRNAME, model_name)
    output_fit_name = name.output_fit
    output_fit_csv = os.path.join(OUTPUT_FIT_DIANAME, output_fit_name)
    model = create_model(which_model=name.which_model)
    train_model(model, name.batch_size, name.epochs, model_saved_path, output_fit_csv)


if __name__ == '__main__':
    F_IMGS = h5py.File("/home/zj/helloworld/kaggle/threat_recognition/hdf5/imgs_preprocessed_200_200_3.h5", "r")
    F_LABELS = F_IMGS["labels"]
    F_FEATURES = h5py.File("/home/zj/helloworld/kaggle/threat_recognition/hdf5/cnn_features_200_trained.h5", "r")
    # F_VGG16_FEATURES = F_FEATURES["VGG16_features"]
    # F_RESNET50_FEATURES = F_FEATURES["ResNet50_features"]
    # F_INCEPTIONV3_FEATURES = F_FEATURES["InceptionV3_features"]
    F_XCEPTION_FEATURES = F_FEATURES["Xception_features"]
    MODEL_SAVE_PATH_DIRNAME = "/home/zj/helloworld/kaggle/threat_recognition/new_model_saved"
    OUTPUT_FIT_DIANAME = "/home/zj/helloworld/kaggle/tr/new_tr/output_fit"
    begin = datetime.now()
    print("开始时间： ", begin)
    __main()
    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))
