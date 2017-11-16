#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_model.py
# @Author: zjj421
# @Date  : 17-8-28
# @Desc  :
from datetime import datetime

import numpy as np
from keras import backend

from online.mymodule.create_model import create_model
from online.mymodule.generate_model import generate_model
from online.mymodule.preprocess_data import preprocess_data_1, zip_train_data, preprocess_data_2, preprocess_data
from online.mymodule.rw_config import get_current_batch_data_root, get_current_model_path, update_config, \
    save_model_weights
from online.mymodule.save_outputs import save_output_fit


def train_model(model, which_model, current_batch_data_root, stage1_labels, x_val, y_val, output_fit_csv, batch_size,
                epochs, model_path=None):
    global new_start
    # 判断模型文件是否存在，存在则载入
    if model_path and new_start:
        generate_model(model_path=model_path, model=model, which_model=None)
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
            hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, shuffle=False,
                             validation_data=(x_val, y_val))
            save_output_fit(hist.history, current_batch_data_root, output_fit_csv)
    else:
        x_train, y_train = preprocess_data(which_model, current_batch_data_root, stage1_labels)
        hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                         validation_data=(x_val, y_val))
        save_output_fit(hist.history, current_batch_data_root, output_fit_csv)
    new_start = False

    return model


def run_model(model, which_model, num_batch, stage1_labels, train_data_root, config_path, model_root, output_fit_csv,
              batch_size, epochs,
              validation_data_root=None):
    # -------------------------------------------------------------------------------------------------------------
    x_val, y_val = preprocess_data(which_model, validation_data_root, stage1_labels)
    # -------------------------------------------------------------------------------------------------------------
    print("验证集获取完毕！")
    print("-" * 100)
    for i in range(num_batch):
        current_batch_data_root = get_current_batch_data_root(train_data_root, config_path, model_root)
        current_model_path = get_current_model_path(config_path)
        print('开始训练"{}"中的数据！\n\n'.format(current_batch_data_root))
        # print('current_model_path: "{}"')
        # 统一输入x_train, y_train的格式，train_model函数中再做相应的格式修改
        model = train_model(model, which_model, current_batch_data_root, stage1_labels, x_val, y_val, output_fit_csv,
                            batch_size, epochs,
                            current_model_path)
        # model = train_model(model, x_train, y_train, current_model_path)
        # 更新配置文件
        update_config(current_batch_data_root, config_path)
        # 保存模型权重
        save_model_weights(model, config_path)
        print('"{}"中的数据训练完毕！\n\n'.format(current_batch_data_root))
    print("*" * 100)
    print("{}个批次的数据训练完毕！".format(num_batch))
    print("*" * 100)


def __main():
    model = create_model(which_model=WHICH_MODEL)
    run_model(model, WHICH_MODEL, NUM_BATCH_TO_TRAIN, STAGE1_LABELS, TRAIN_DATA_ROOT, CONFIG_PATH, MODEL_ROOT,
              OUTPUT_FIT_CSV, BATCH_SIZE, EPOCHS, VALIDATION_DATA_ROOT)


if __name__ == '__main__':
    # －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
    # 以下内容需要手动指定
    WHICH_MODEL = 5
    NTH_WHICH_MODEL = 6
    NUM_VAL = 100
    BATCH_SIZE = 4
    EPOCHS = 12
    NUM_BATCH_TO_TRAIN = 5
    # －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

    file = "model{which_model}_{nth_which_model}_{num_val}val_{batch_size}_{epochs}".format(
        which_model=WHICH_MODEL,
        nth_which_model=NTH_WHICH_MODEL,
        num_val=NUM_VAL,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    OUTPUT_FIT_CSV = "/home/zj/helloworld/kaggle/tr/output_fit/output_fit_{file}.csv".format(file=file)
    OUTPUT_FIT = "/home/zj/helloworld/kaggle/tr/output_fit/output_fit_{file}.json".format(file=file)
    CONFIG_PATH = "/home/zj/helloworld/kaggle/tr/config/{file}_config.json".format(file=file)
    MODEL_ROOT = "/home/zj/helloworld/kaggle/threat_recognition/models_saved/{file}".format(file=file)
    TRAIN_DATA_ROOT = "/home/zj/helloworld/kaggle/threat_recognition/syblink_stage1_aps/{file}/train_data_root".format(
        file=file)
    VALIDATION_DATA_ROOT = "/home/zj/helloworld/kaggle/threat_recognition/syblink_stage1_aps/{file}/validation_data_root".format(
        file=file)
    # if not os.path.exists(MODEL_ROOT):
    #     print("请重新指定 '{}'!!!".format(MODEL_ROOT))
    #     exit()
    new_start = True
    STAGE1_LABELS = "/media/zj/study/kaggle/stage1_labels.csv"
    begin = datetime.now()
    print("开始时间： ", begin)
    # 0 = test, 1 = train
    backend.set_learning_phase(1)
    __main()
    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))
