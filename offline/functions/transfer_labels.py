#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : transfer_labels.py
# @Author: zjj421
# @Date  : 17-11-9
# @Desc  :
import h5py
import numpy as np


def __main():
    f_labels_1 = F_IMGS_1["labels"]
    f_labels_2 = F_FEATURES.create_group("labels")

    for key in f_labels_1.keys():
        print(key)
        value = f_labels_1[key].value
        value = np.asarray(value)
        f_labels_2.create_dataset(str(key), data=value)
        # print(f[key].shape)
        print("-" * 15)
    print("Done")


if __name__ == '__main__':
    F_IMGS_1 = h5py.File(
        "/home/zj/helloworld/kaggle/threat_recognition/hdf5/imgs_preprocessed_200_200_3.h5",
        "r")
    F_FEATURES = h5py.File(
        "/home/zj/helloworld/kaggle/threat_recognition/hdf5/cnn_features_200_trained.h5",
        "a")
    __main()
