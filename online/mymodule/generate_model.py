#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : generate_model.py
# @Author: zjj421
# @Date  : 17-9-23
# @Desc  :
import os

from online.mymodule.create_model import create_model


def generate_model(model_path, model, which_model):
    if model and (not which_model):
        model = model
    elif (not model) and which_model:
        model = create_model(which_model)
    else:
        model = None
        print("请指定model！")
        exit()
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print("模型权重'{}'导入成功".format(model_path))
    else:
        print("请重新指定model_path!")
        exit()
    return model

