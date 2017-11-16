#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict_model.py
# @Author: zjj421
# @Date  : 17-9-15
# @Desc  :

import numpy as np
import pandas as pd
from keras import backend
from mymodule.generate_model import generate_model

from online.mymodule.preprocess_data import preprocess_data

angle_01 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
angle_02 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
angle_03 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
angle_04 = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
angle_05 = [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
angle_06 = [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
angle_07 = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
angle_08 = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
angle_09 = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
angle_10 = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
angle_11 = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
angle_12 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
angle_13 = [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
angle_14 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]
angle_15 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
angle_16 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
angles = [angle_01, angle_02, angle_03, angle_04, angle_05, angle_06, angle_07, angle_08, angle_09, angle_10, angle_11,
          angle_12, angle_13, angle_14, angle_15, angle_16]


def predict_model(which_model, model_path, test_data_root, batch_size):
    model = generate_model(model_path=model_path, model=None, which_model=which_model)
    x_test, subject_id_set_list = preprocess_data(which_model, test_data_root)
    Id = []
    for i in subject_id_set_list:
        for n in range(1, 18):
            id = "{}_Zone{}".format(i, n)
            Id.append(id)
    print("总共有{}个id".format(len(Id)))
    # 预测时的batch_size对结果有影响
    predict_values = model.predict(x_test, batch_size=batch_size, verbose=1)
    if which_model == 1:
        num_angles = 16
        num_zones = 17
        num_test_data = 100
        predict_values_set = []
        idx = 0
        for i in range(num_test_data):
            sub_predict_values_set = []
            arr = predict_values[idx:idx + num_angles] * angles
            idx = idx + num_angles
            for j in range(num_zones):
                sum = 0
                count = 0
                for k in range(num_angles):
                    if arr[k][j]:
                        count = count + 1
                        sum = sum + arr[k][j]
                mean = sum / count
                sub_predict_values_set.append(mean)
            predict_values_set.append(sub_predict_values_set)
        predict_values = np.asarray(predict_values_set, dtype=np.float32)
    print(predict_values.shape)
    predict_values = predict_values.reshape(-1)
    dataframe = pd.DataFrame({"Id": Id, "Probability": predict_values})
    # ------------------------------------------------------------------------------------------------------------------
    # 手动修改
    dataframe.to_csv(
        OUTPUT_SUBMISSION,
        index=False, sep=",")
    # －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
    print(predict_values.shape)
    # for i in predict_values:
    #     print(i)
    # data = pd.read_csv(predict_values)
    # data.to_csv("/tmp/test001.csv")


if __name__ == '__main__':
    WHICH_MODEL = 5
    BATCH_SIZE = 4
    MODEL_PATH = "/home/zj/helloworld/kaggle/threat_recognition/models_saved/model5_6_100val_4_12/tsa-dir-003-No003.h5"
    OUTPUT_SUBMISSION = "/home/zj/helloworld/kaggle/tr/stage1_submissions/stage1_submission_5_6_100_4_12_4pbs_dir003.csv"
    TEST_DATA_ROOT = "/home/zj/helloworld/kaggle/threat_recognition/syblink_stage1_aps/test_data_root"
    backend.set_learning_phase(0)
    predict_model(WHICH_MODEL, MODEL_PATH, TEST_DATA_ROOT, BATCH_SIZE)
