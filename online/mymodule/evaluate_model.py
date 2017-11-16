#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate_model.py
# @Author: zjj421
# @Date  : 17-9-19
# @Desc  :

import pandas as pd
from keras import backend

from online.mymodule.preprocess_data import preprocess_data
from .generate_model import generate_model


def evaluate_model(which_model, model_path, validation_data_root, stage1_labels, batch_size):
    model = generate_model(model_path=model_path, model=None, which_model=which_model)
    x_val, y_val = preprocess_data(which_model, validation_data_root, stage1_labels)
    df = pd.DataFrame()
    for i in range(1, 10):
        val_loss = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        print("batch_size= {}, val_loss = {}.".format(batch_size, val_loss))
        batch_size = batch_size + 1
        df["validation_batch_size"].append(i)
        df["val_loss"].append(val_loss)
    df.to_csv(
        OUTPUT_VAL_LOSS,
        index=False, sep=",")


if __name__ == '__main__':
    WHICH_MODEL = 5
    BATCH_SIZE = 3
    MODEL_PATH = "/home/zj/helloworld/kaggle/threat_recognition/models_saved/model5_5_100val_6_12/tsa-dir-005-No005.h5"
    OUTPUT_VAL_LOSS = "/home/zj/helloworld/kaggle/tr/output_evaluate/output_evaluate_2_1_100val_3_12.csv"
    VALIDATION_DATA_ROOT = "/home/zj/helloworld/kaggle/threat_recognition/syblink_stage1_aps/model5_5_100val_6_12/validation_data_root"
    STAGE1_LABELS = "/media/zj/study/kaggle/stage1_labels.csv"
    backend.set_learning_phase(0)
    evaluate_model(WHICH_MODEL, MODEL_PATH, VALIDATION_DATA_ROOT, STAGE1_LABELS, BATCH_SIZE)
