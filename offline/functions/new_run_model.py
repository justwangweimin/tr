#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : new_run_model.py
# @Author: zjj421
# @Date  : 17-11-9
# @Desc  :
from offline.new_mymodules.baserunmodel import BaseRunModel


def __main():
    run_model = BaseRunModel(which_model=3002,
                             cnn_name="ResNet50",
                             img_height=200,
                             num_val=100,
                             batch_size=3,
                             epochs=12,
                             flag="1",
                             # model_path="/home/zj/dataShare/Linux_win10/helloworld/kaggle/threat_recognition/new_model_saved/model_1001_InceptionV3_200_100val_3_120_flag_1.h5",
                             predict_batch_size=3,
                             f_features_trained_path=F_FEATURES_TRAINED_PATH,
                             f_features_tested_path=F_FEATURES_TESTED_PATH
                             )
    x_train, y_train = run_model.prepare_train_data()
    x_val, y_val = run_model.prepare_val_data()
    # x_val, y_val = run_model.prepare_test_data_2()

    run_model.train_model(x_train, y_train, x_val, y_val)
    x_test, ids = run_model.prepare_test_data()
    run_model.predict_model(x_test, ids, save_output=True)

    x_test2, y_test2 = run_model.prepare_test_data_2()
    run_model.evaluate_model(x_test2, y_test2)


if __name__ == '__main__':
    F_FEATURES_TRAINED_PATH = "/home/zj/helloworld/kaggle/threat_recognition/hdf5/cnn_features_200_trained.h5"
    F_FEATURES_TESTED_PATH = "/home/zj/helloworld/kaggle/threat_recognition/hdf5/cnn_features_200_tested.h5"

    __main()
