#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate_model.py
# @Author: zjj421
# @Date  : 17-9-19
# @Desc  :
import h5py
import pandas as pd
from keras.losses import binary_crossentropy

from offline.functions.imgs_preprocessed.img_preprocessed_save2hdf5 import get_single_label
from offline.functions.run_model import prepare_data, get_keys
from offline.new_mymodules.my_model import MyModel


def get_subject_ids(labels):
    df = pd.read_csv(labels)
    df['Subject'], df['Zone'] = df['Id'].str.split('_', 1).str
    ids = set(df["Subject"])
    # print(type(ids))
    # print(ids)
    # print(len(ids))
    return ids


def save_labels2hdf5(labels_group, labels):
    subject_ids_list = get_subject_ids(labels)
    for subject_id in subject_ids_list:
        zone_label_list = get_single_label(labels, subject_id)
        # 每个dataset包含一个人的一组标签
        # 默认压缩等级为４
        labels_group.create_dataset(str(subject_id), data=zone_label_list, compression="gzip",
                                    compression_opts=4)
    print("Done")


def __main():
    mymodel = MyModel(which_model=1005,
                      cnn_name="VGG19",
                      img_height=200,
                      num_val=0,
                      batch_size=3,
                      epochs=50,
                      flag="1",
                      model_path="/home/zj/helloworld/kaggle/tr/offline/results/model_1005_VGG19_200_0val_3_60_flag_test/model_1005_VGG19_200_0val_3_60_flag_test.h5",
                      predict_batch_size=3)
    cnn_name = mymodel.cnn_name
    num_val = mymodel.num_val
    test_keys = get_keys(F_FEATURES_TESTED, cnn_name, num_split=num_val)
    x_test, y_test = prepare_data(F_FEATURES_TESTED, cnn_name, test_keys, f_labels=LABELS_GROUP)
    mymodel.evaluate_model(x_test, y_test)


if __name__ == '__main__':
    F_FEATURES_TESTED = h5py.File("/home/zj/helloworld/kaggle/threat_recognition/hdf5/cnn_features_200_tested.h5", "r")

    # LABELS = "/home/zj/桌面/test_labels.csv"
    f_test_labels = "test_data.h5"
    F = h5py.File(f_test_labels, "r")
    try:
        LABELS_GROUP = F.create_group("test_labels")
    except:
        LABELS_GROUP = F["test_labels"]

    __main()
    # save_labels2hdf5(LABELS_GROUP, LABELS)