#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : basedata.py
# @Author: zjj421
# @Date  : 17-11-9
# @Desc  :
import h5py
import numpy as np
import os


class BaseData(object):
    def __init__(self, which_model,
                 cnn_name,
                 f_features_trained_path,
                 f_features_tested_path,
                 num_val=0):
        self.which_model = which_model
        self.cnn_name = cnn_name
        self.num_val = num_val
        self.f_features_trained_path = f_features_trained_path
        self.f_features_tested_path = f_features_tested_path

    @property
    def f_features_labels_trained(self):
        if not os.path.exists(self.f_features_trained_path):
            print("'{}'不存在,请重新指定!".format(self.f_features_trained_path))
        f_trained = h5py.File(self.f_features_trained_path, "r")
        f_features_trained = f_trained[self.cnn_name + "_features"]
        f_labels_trained = f_trained["labels"]
        return [f_features_trained, f_labels_trained]

    @property
    def f_features_tested(self):
        if not os.path.exists(self.f_features_tested_path):
            print("'{}'不存在,请重新指定!".format(self.f_features_tested_path))
        f_features_tested = h5py.File(self.f_features_tested_path, "r")
        f_features_tested = f_features_tested[self.cnn_name + "_features"]
        return f_features_tested

    @property
    def f_labels_tested(self):
        f_labels_tested = h5py.File("/home/zj/helloworld/kaggle/tr/offline/test_data.h5", "r")
        f_labels_tested = f_labels_tested["test_labels"]
        return f_labels_tested

    @property
    def train_val_keys(self):
        if self.num_val > 0:
            train_keys, val_keys = self._get_keys(f_features=self.f_features_labels_trained[0], num_val=self.num_val)
        else:
            train_keys = self._get_keys(f_features=self.f_features_labels_trained[0], num_val=self.num_val)
            val_keys = None
        return [train_keys, val_keys]

    # test_keys的num_val一直为0
    @property
    def test_keys(self):
        test_keys = self._get_keys(self.f_features_tested, num_val=0)
        return test_keys

    @property
    def train_keys(self):
        return self.train_val_keys[0]

    @property
    def val_keys(self):
        return self.train_val_keys[1]

    def prepare_data(self, f_features, keys, f_labels):
        # 没有val_keys的时候返回None
        if not keys:
            return None, None
        cnn_features = []
        labels = []
        ids = []
        for key in keys:
            cnn_feature = f_features[key].value
            cnn_features.append(cnn_feature)
            if f_labels:
                label = f_labels[key].value
                labels.append(label)
            else:
                for n in range(1, 18):
                    id = "{}_Zone{}".format(key, n)
                    ids.append(id)
        cnn_features = np.asarray(cnn_features, dtype=np.float32)
        labels = np.asarray(labels)
        which_shape = self.which_model // 1000
        # 根据模型决定输出数据的格式
        if which_shape == 1:
            # 帧数写死了,为16帧
            # 16个input
            dim_1 = cnn_features.shape[-1]
            cnn_features = np.array_split(cnn_features, 16, axis=1)
            for i, v in enumerate(cnn_features):
                v = v.reshape(-1, dim_1)
                cnn_features[i] = v
        elif which_shape == 2:
            # 1个input
            pass
        else:
            print("请修改basedata.py")

        if f_labels:
            return cnn_features, labels
        else:
            return cnn_features, ids

    def prepare_train_data(self):
        x_train, y_train = self.prepare_data(self.f_features_labels_trained[0], self.train_keys,
                                             self.f_features_labels_trained[1])
        return x_train, y_train

    def prepare_val_data(self):
        x_val, y_val = self.prepare_data(self.f_features_labels_trained[0], self.val_keys,
                                         self.f_features_labels_trained[1])
        return x_val, y_val

    def prepare_test_data(self):
        x_test, ids = self.prepare_data(self.f_features_tested, self.test_keys,
                                        f_labels=None)
        return x_test, ids

    def prepare_test_data_2(self):
        x_test, y_test = self.prepare_data(self.f_features_tested, self.test_keys,
                                           f_labels=self.f_labels_tested)
        return x_test, y_test

    def _get_keys(self, f_features, num_val):

        keys = []
        for key in f_features.keys():
            keys.append(key)
        np.random.shuffle(keys)
        if num_val > 0:
            validation_keys = keys[:num_val]
            train_keys = keys[num_val:]
            return train_keys, validation_keys
        else:
            return keys


def __main():
    pass


if __name__ == '__main__':
    __main()
