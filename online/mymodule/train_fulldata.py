#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train_fulldata.py
# @Author: zjj421
# @Date  : 17-9-13
# @Desc  :
import os

import numpy as np
from keras import backend
from mymodule.generate_model import generate_model
from mymodule.save_outputs import save_output_fit

from online.mymodule.preprocess_data import preprocess_data_1, preprocess_data_2, zip_train_data, preprocess_data


def train_full_data_model(model, which_model, current_batch_data_root, stage1_labels, output_fit_csv, batch_size, epochs):
    # ----------------------------------------------------------------------------------------------------------
    if which_model == 1:
        # model_1
        x_train, y_train = preprocess_data_1(current_batch_data_root, stage1_labels)
        train_data = zip_train_data(x_train, y_train)
        for e in range(epochs):
            # 打乱
            np.random.shuffle(train_data)
            print("-" * 100)
            print("Epoch: {}/{}".format((e + 1), EPOCHS))
            x_train = []
            y_train = []
            for i in train_data:
                x_train.append(i[0])
                y_train.append(i[1])
            print("第　{}　次打乱数据".format(e + 1))
            x_train, y_train = preprocess_data_2(x_train, y_train)
            print("x_train.shape: ", x_train.shape)
            print("y_train.shape: ", y_train.shape)
            hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, shuffle=False)
            save_output_fit(hist.history, current_batch_data_root, output_fit_csv)
    else:
        x_train, y_train = preprocess_data(which_model, current_batch_data_root, stage1_labels)
        hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
        save_output_fit(hist.history, current_batch_data_root, output_fit_csv)
    return model


def save_model_weights(model, model_path):
    dirname = os.path.dirname(model_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    model.save_weights(model_path)


def __main():
    model = generate_model(model_path=MODEL_PATH, model=None, which_model=WHICH_MODEL)
    model = train_full_data_model(model, WHICH_MODEL, TRAIN_DATA_DIRNAME, STAGE1_LABELS, OUTPUT_FIT_CSV. BATCH_SIZE, EPOCHS)
    save_model_weights(model, NEW_MODEL_PATH)


if __name__ == '__main__':
    WHICH_MODEL = 5
    NTH_WHICH_MODEL = 5
    NUM_VAL = 0
    BATCH_SIZE = 6
    EPOCHS = 12
    dst_file = "model{which_model}_{nth_which_model}_{num_val}val_{batch_size}_{epochs}".format(
        which_model=WHICH_MODEL,
        nth_which_model=NTH_WHICH_MODEL,
        num_val=NUM_VAL,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    src_file = "model{which_model}_{nth_which_model}_{num_val}val_{batch_size}_{epochs}".format(
        which_model=WHICH_MODEL,
        nth_which_model=NTH_WHICH_MODEL,
        num_val=100,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    # MODEL_PATH = "/home/zj/helloworld/kaggle/threat_recognition/models_saved/model_4_2_0val_repeat.h5"
    MODEL_PATH = "/home/zj/helloworld/kaggle/threat_recognition/models_saved/{file}/tsa-dir-005-No005.h5".format(
        file=src_file)
    TRAIN_DATA_DIRNAME = "/home/zj/helloworld/kaggle/threat_recognition/syblink_stage1_aps/{file}/validation_data_root".format(
        file=src_file)
    # TRAIN_DATA_DIRNAME = "/home/zj/helloworld/kaggle/threat_recognition/syblink_stage1_aps/model4_2_100val_3_12/validation_data_root"
    NEW_MODEL_PATH = "/home/zj/helloworld/kaggle/threat_recognition/models_saved/{file}.h5".format(file=dst_file)
    # NEW_MODEL_PATH = "/home/zj/helloworld/kaggle/threat_recognition/models_saved/model_4_2_0val_repeat_1.h5"
    OUTPUT_FIT = "/home/zj/helloworld/kaggle/tr/output_fit/output_fit_{file}.json".format(file=dst_file)
    # OUTPUT_FIT = "/home/zj/helloworld/kaggle/threat_recognition/output_fit/output_fit_model_4_2_0val_repeat_1.json"
    OUTPUT_FIT_CSV = "/home/zj/helloworld/kaggle/tr/output_fit/output_fit_{file}.csv".format(
        file=dst_file)
    # OUTPUT_FIT_CSV = "/home/zj/helloworld/kaggle/threat_recognition/output_fit/output_fit_model_4_2_0val_repeat_1.csv"

    STAGE1_LABELS = "/media/zj/study/kaggle/stage1_labels.csv"

    backend.set_learning_phase(1)
    __main()
