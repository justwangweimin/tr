#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : my_model.py
# @Author: zjj421
# @Date  : 17-10-13
# @Desc  :
import os

from offline.new_mymodules.basemodel import BaseModel


class MyModel(BaseModel):
    def __init__(self, which_model=None,
                 cnn_name=None,
                 img_height=200,
                 num_val=0,
                 batch_size=3,
                 epochs=12,
                 model_path=None,
                 predict_batch_size=3,
                 flag="1"):
        self.cnn_name = cnn_name
        self.img_height = img_height
        super().__init__(which_model=which_model,
                         num_val=num_val,
                         batch_size=batch_size,
                         epochs=epochs,
                         model_path=model_path,
                         predict_batch_size=predict_batch_size,
                         flag=flag)


    # 只读属性
    @property
    def directory_name(self):
        if not self.model_path:
            name = "model_{which_model}_{cnn_name}_{img_height}_{num_val}val_{batch_size}_{epochs}_flag_{flag}".format(
                which_model=self.which_model,
                cnn_name=self.cnn_name,
                img_height=self.img_height,
                num_val=self.num_val,
                batch_size=self.batch_size,
                epochs=self.epochs,
                flag=self.flag)
        else:
            _name = os.path.splitext(os.path.basename(self.model_path))[0].split("_flag_", 1)[0]
            name = "{}_flag_{}".format(_name, self.flag)
        return name
