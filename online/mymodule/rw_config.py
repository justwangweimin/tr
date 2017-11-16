#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : rw_config.py
# @Author: zjj421
# @Date  : 17-9-5
# @Desc  :
import inspect
import json

import os


# 获取传值给变量var的变量的变量名。
def get_var_name(var):
    # get back the name of variables
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


# 更新配置文件
def update_config(current_batch_data_root, config_path):
    # 1. 必须首先设置last_batch_data_root，写进配置文件。
    last_batch_data_root = current_batch_data_root
    set_last_batch_data_root(last_batch_data_root, config_path)
    # 2. 同时也更新num_models
    set_last_model_path(config_path)
    # 3.
    set_model_file_list(config_path)


def write_config(var, config_path):
    key = str(get_var_name(var))
    with open(config_path, "r") as f:
        pyobj = json.load(f)
        pyobj[key] = var
    with open(config_path, "w") as f:
        json.dump(pyobj, f, indent=4)


def read_config(config_path):
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        pyobj = json.load(f)
    return pyobj


def generate_data_dir_list(data_root):
    data_dir_list = []
    for root, dirs, files in os.walk(data_root):
        for dir in dirs:
            data_dirname = dir
            data_dir_list.append(data_dirname)
    data_dir_list.sort()
    return data_dir_list


def config_initial(data_root, model_root, config_path):
    pydict = {
        "data_root": data_root,
        "model_root": model_root,
        "data_dir_list": [],
        "model_file_list": [],
        "last_batch_data_root": "",
        "last_model_path": "",
        "num_models": 0,
    }

    data_dir_list = generate_data_dir_list(data_root)
    pydict["data_dir_list"] = data_dir_list

    with open(config_path, "w") as f:
        json.dump(pydict, f, indent=4)


def get_last_batch_data_root(config_path):
    last_batch_data_root = read_config(config_path)["last_batch_data_root"]

    return last_batch_data_root


def set_last_batch_data_root(last_batch_data_root, config_path):
    write_config(last_batch_data_root, config_path)


def set_last_model_path(config_path):
    num_models = get_num_models(config_path)
    num_models = num_models + 1

    last_batch_data_root = get_last_batch_data_root(config_path)
    last_batch_data_root_dir = os.path.split(last_batch_data_root)[1]
    model_file = "tsa-{:0>3}-No{:0>3}.h5".format(last_batch_data_root_dir, num_models)
    model_root = get_model_root(config_path)
    last_model_path = os.path.join(model_root, model_file)
    write_config(last_model_path, config_path)
    write_config(num_models, config_path)


def get_last_model_path(config_path):
    pydict = read_config(config_path)
    last_model_path = pydict["last_model_path"]

    return last_model_path


def get_model_file_list(config_path):
    pydict = read_config(config_path)
    model_file_list = pydict["model_file_list"]
    return model_file_list


def set_model_file_list(config_path):
    last_model_path = get_last_model_path(config_path)
    last_model_file = os.path.split(last_model_path)[1]
    model_file_list = get_model_file_list(config_path)
    model_file_list.append(last_model_file)
    write_config(model_file_list, config_path)


def get_batch_data_root_list(data_root, config_path, model_root):
    if not os.path.exists(config_path):
        dirname = os.path.dirname(config_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        config_initial(data_root, model_root, config_path)
    data_dir_list = read_config(config_path)["data_dir_list"]
    batch_data_root_list = list(
        map(lambda data_dirname: os.path.join(data_root, data_dirname), data_dir_list))
    return batch_data_root_list


def get_current_batch_data_root(train_data_root, config_path, model_root):
    # 同时初始化配置文件
    batch_data_root_list = get_batch_data_root_list(train_data_root, config_path, model_root)
    # 从配置文件中读取上次运行的目录名（全名）
    last_batch_data_root = get_last_batch_data_root(config_path)
    if not last_batch_data_root:
        last_batch_data_root = batch_data_root_list[0]
        current_batch_data_root = last_batch_data_root
    else:
        current_batch_data_root = batch_data_root_list[batch_data_root_list.index(last_batch_data_root) + 1]
    return current_batch_data_root


def get_current_model_path(config_path):
    last_model_path = get_last_model_path(config_path)
    # 上次保存的模型权重导入本次训练的模型中
    current_model_path = last_model_path
    return current_model_path


def get_num_models(config_path):
    pydict = read_config(config_path)
    num_models = pydict["num_models"]
    return num_models


def get_model_root(config_path):
    pydict = read_config(config_path)
    model_root = pydict["model_root"]
    return model_root

def save_model_weights(model, config_path):
    model_path = get_last_model_path(config_path)
    dirname = os.path.dirname(model_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    model.save_weights(model_path)

if __name__ == '__main__':
    pass
