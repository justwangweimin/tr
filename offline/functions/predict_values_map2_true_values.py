#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict_values_map2_true_values.py
# @Author: zjj421
# @Date  : 17-10-30
# @Desc  :
import os
from datetime import datetime

from new_tr.functions.get_threshold.sort_zone_id import sort_zone_id
from new_tr.functions.run_model import save_runtime

from offline.new_mymodules import BaseModel


class MyModel2(BaseModel):
    def __init__(self, which_model=None,
                 num_val=0,
                 batch_size=3,
                 epochs=12,
                 flag="1",
                 model_path=None,
                 predict_batch_size=3,
                 predict_values_filename=None):
        self.predict_values_filename = predict_values_filename
        super().__init__(which_model=which_model,
                         num_val=num_val,
                         batch_size=batch_size,
                         epochs=epochs,
                         model_path=model_path,
                         predict_batch_size=predict_batch_size,
                         flag=flag)

    @property
    def _name(self):
        if not self.model_path:
            _name = os.path.splitext(self.predict_values_filename)[0].split("_submission_", 1)[1]
            name = "{}_prdt2t_{num_val}val_{batch_size}_{epochs}_flag2_{flag}".format(
                _name,
                num_val=self.num_val,
                batch_size=self.batch_size,
                epochs=self.epochs,
                flag=self.flag,
            )
        else:
            _name = os.path.splitext(os.path.basename(self.model_path))[0].split("_flag2_", 1)[0]
            name = "{}_flag2_{}".format(_name, self.flag)
        return name


def get_values(csvfile, retu_ids=False):
    df = sort_zone_id(csvfile, save_ouput=False)
    values = df["Probability"].reshape(-1, 17)
    print("查看顺序是否一样：")
    print(df["Id"])
    print("-" * 100)
    print("values.shape", values.shape)
    if retu_ids:
        return values, df["Id"]
    return values


def __main():
    begin = datetime.now()
    print("开始时间： ", begin)

    predict_values = get_values(PREDICT_VALUES_CSV)
    true_values = get_values(TRUE_VALUES_CSV)
    mymodel = MyModel2(
        which_model=9001,
        num_val=0,
        batch_size=3,
        epochs=1000,
        flag="predict_values_2",
        model_path="/home/zj/helloworld/kaggle/threat_recognition/new_model_saved/predict2true_model/model_1002_VGG19_200_0val_3_50_flag_train_data_predict_5pbs_prdt2t_0val_3_1000_flag2_1.h5",
        predict_batch_size=3,
        predict_values_filename=PREDICT_VALUES_CSV,
    )

    # 可写进类里面
    model = mymodel.generate_model()
    # model = mymodel.train_model(model, x_train=predict_values, y_train=true_values)
    # x_test, ids = get_values(TEST_VALUES_CSV, retu_ids=True)
    x_test, ids = get_values(PREDICT_VALUES_CSV, retu_ids=True)
    print(type(x_test))
    print("*" * 100)
    print(type(ids))
    mymodel.predict_model(model, x_test, ids)

    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))
    try:
        save_runtime(mymodel, time)
    except:
        print("未训练模型，没有output文件输出！！")


if __name__ == '__main__':
    MODEL_SAVED_DIRNAME = "/home/zj/helloworld/kaggle/threat_recognition/new_model_saved/predict2true_model"
    OUTPUT_FIT_DIANAME = "/home/zj/helloworld/kaggle/tr/new_tr/output_fit/output_fit_predict2true"
    STAGE1_SUBMISSION_DIRNAME = "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/predict2true"
    PREDICT_VALUES_CSV = "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/stage1_submission_model_1002_VGG19_200_0val_3_50_flag_train_data_predict_5pbs.csv"
    TRUE_VALUES_CSV = "/home/zj/helloworld/kaggle/tr/new_tr/functions/get_threshold/sorted_stage1_labels.csv"
    TEST_VALUES_CSV = "/home/zj/helloworld/kaggle/tr/new_tr/submissions/stage1_submissions/stage1_submission_model_1002_VGG19_200_0val_3_50_flag_1_5pbs.csv"
    __main()
