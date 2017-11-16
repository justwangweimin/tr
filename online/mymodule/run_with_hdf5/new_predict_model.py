#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : new_predict_model.py
# @Author: zjj421
# @Date  : 17-10-12
# @Desc  :
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from online.mymodule import generate_model


def predict_model(model, batch_size):
    Id = []
    features = []
    for i, key in enumerate(F_RESNET50_FEATURES.keys(), start=1):
        for n in range(1, 18):
            id = "{}_Zone{}".format(key, n)
            Id.append(id)
        resnet50_feature = F_RESNET50_FEATURES[key].value
        print(i)
        features.append(resnet50_feature)
    features = np.asarray(features, dtype=np.float32)
    features_list = np.array_split(features, 16, axis=1)
    dim_1 = features.shape[-1]
    for j, x in enumerate(features_list):
        x = x.reshape(-1, dim_1)
        features_list[j] = x
    predict_values = model.predict(features_list, batch_size=batch_size, verbose=1)
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


def __main():
    model = generate_model(model_path=MODEL_PATH, model=None, which_model=WHICH_MODEL)
    predict_model(model=model, batch_size=5)


if __name__ == '__main__':
    MODEL_PATH = "/home/zj/helloworld/kaggle/threat_recognition/new_model_saved/model_1001_xce_200_0val_3_120_no_1.h5"
    WHICH_MODEL = 1001
    F_FEATURES = h5py.File("/home/zj/helloworld/kaggle/threat_recognition/hdf5/cnn_features_200_tested.h5", "r")
    F_RESNET50_FEATURES = F_FEATURES["Xception_features"]
    OUTPUT_SUBMISSION = "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/output_submission_model_" \
                        "1001_xce_200_0val_3_120_no_1_5pbs.csv"
    begin = datetime.now()
    print("开始时间： ", begin)
    __main()
    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))
