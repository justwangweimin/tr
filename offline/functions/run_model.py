#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_model.py
# @Author: zjj421
# @Date  : 17-10-13
# @Desc  :
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from offline.new_mymodules.my_model import MyModel


def get_keys(f_features, cnn_name, num_split=100):
    f_features = f_features[cnn_name + "_features"]
    keys = []
    for key in f_features.keys():
        keys.append(key)
    np.random.shuffle(keys)
    if num_split > 0:
        validation_keys = keys[:num_split]
        train_keys = keys[num_split:]
        return train_keys, validation_keys
    else:
        return keys


def prepare_data(f_features, cnn_name, keys, f_labels):
    if not keys:
        return None, None
    f_features = f_features[cnn_name + "_features"]
    cnn_features = []
    labels = []
    ids = []
    for key in keys:
        cnn_feature = f_features[key].value
        cnn_features.append(cnn_feature)
        if f_labels:
            label = f_labels[key].value
            labels.append(label)
        else:
            for n in range(1, 18):
                id = "{}_Zone{}".format(key, n)
                ids.append(id)
    cnn_features = np.asarray(cnn_features, dtype=np.float32)
    dim_1 = cnn_features.shape[-1]
    cnn_features_list = np.array_split(cnn_features, 16, axis=1)
    for i, v in enumerate(cnn_features_list):
        v = v.reshape(-1, dim_1)
        cnn_features_list[i] = v
    labels = np.asarray(labels)
    if f_labels:
        return cnn_features_list, labels
    else:
        return cnn_features_list, ids


def save_keys(train_keys, val_keys, save_dirname, mymodel_name):
    if not os.path.exists(save_dirname):
        # makedirs可生成多级递归目录
        os.makedirs(save_dirname)
    csv_name = "train_val_ids_{}.csv".format(mymodel_name)
    csv_path = os.path.join(save_dirname, csv_name)
    # if os.path.exists(csv_path):
    #     print("请重新指定MyModel参数，{} 已存在！".format(csv_path))
    #     exit()
    # print(len(train_keys))
    # print(len(val_keys))
    train_keys = Series(train_keys)
    val_keys = Series(val_keys)
    data = {"train_ids": train_keys,
            "val_ids": val_keys}
    df = DataFrame(data)
    df.to_csv(csv_path,
              index=False,
              sep=",")


def save_runtime(obj, runtime):
    output_fit_csv = os.path.join(obj.root, obj.output_fit_basename)
    df = pd.read_csv(output_fit_csv)
    df["runtime"] = runtime
    df.to_csv(output_fit_csv, index=False)
    print("运行时间已写进csv文件！")


def __main():
    begin = datetime.now()
    print("开始时间： ", begin)
    mymodel = MyModel(which_model=3001,
                      cnn_name="VGG19",
                      img_height=200,
                      num_val=100,
                      batch_size=3,
                      epochs=60,
                      flag="1",
                      # model_path="/home/zj/dataShare/Linux_win10/helloworld/kaggle/threat_recognition/new_model_saved/model_1001_InceptionV3_200_100val_3_120_flag_1.h5",
                      predict_batch_size=3)
    cnn_name = mymodel.cnn_name
    num_val = mymodel.num_val
    if num_val > 0:
        train_keys, val_keys = get_keys(F_FEATURES_TRAINED, cnn_name, num_split=num_val)
    else:
        train_keys = get_keys(F_FEATURES_TRAINED, cnn_name, num_split=num_val)
        val_keys = None
    save_keys(train_keys, val_keys, save_dirname=mymodel.root, mymodel_name=mymodel.directory_name)
    x_train, y_train = prepare_data(F_FEATURES_TRAINED, cnn_name, train_keys, f_labels=F_LABELS)
    x_val, y_val = prepare_data(F_FEATURES_TRAINED, cnn_name, val_keys, f_labels=F_LABELS)
    mymodel.train_model(x_train, y_train, x_val, y_val)

    test_keys = get_keys(F_FEATURES_TESTED, cnn_name, num_split=0)
    x_test, ids = prepare_data(F_FEATURES_TESTED, cnn_name, test_keys, f_labels=None)
    print(type(ids))
    mymodel.predict_model(x_test, ids, save_output=True)

    end = datetime.now()
    time = (end - begin).seconds
    save_runtime(mymodel, time)
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))


if __name__ == '__main__':
    F_IMGS = h5py.File("/home/zj/helloworld/kaggle/threat_recognition/hdf5/imgs_preprocessed_200_200_3.h5", "r")
    F_LABELS = F_IMGS["labels"]
    F_FEATURES_TRAINED = h5py.File("/home/zj/helloworld/kaggle/threat_recognition/hdf5/cnn_features_200_trained.h5",
                                   "r")
    F_FEATURES_TESTED = h5py.File("/home/zj/helloworld/kaggle/threat_recognition/hdf5/cnn_features_200_tested.h5", "r")
    TRAIN_VAL_IDS_SAVED_DIRNAME = "/home/zj/helloworld/kaggle/tr/new_tr/train_val_ids"

    __main()
