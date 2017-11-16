#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : myown_layers.py
# @Author: zjj421
# @Date  : 17-9-26
# @Desc  :

import numpy as np
from keras.engine import Layer
from keras.layers import Lambda, Dense, initializers
from keras.models import Sequential
import keras.backend as K


class MyLayer1(Layer):
    def __init__(self, output_dim, **kw):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kw)

    def build(self, input_shape):
        input_dim = input_shape[1]
        assert (input_dim == self.output_dim)
        inital_SCALER = np.ones((input_dim,)) * 1000
        self.SCALER = K.variable(inital_SCALER)
        self.trainable_weights = [self.SCALER]
        super(MyLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # return x - K.mean(x,axis=1,keepdims=True)
        x *= self.SCALER
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    # 定义权重，可训练的权应该在这里被加入列表'self.trainable_weights中'
    # build方法必须设置self.built = True，可通过调用super([layer], self).build()实现
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    # 定义层功能
    def call(self, x):
        return K.dot(x, self.kernel)

    # 如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断。
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def sub_mean(x):
    x -= K.mean(x, axis=1, keepdims=True)
    return x


def get_submean_model():
    model = Sequential()
    model.add(Dense(7, input_dim=7, use_bias=False))
    model.add(Lambda(sub_mean, output_shape=lambda input_shape: input_shape))
    model.compile(optimizer='rmsprop', loss='mse')
    return model


if __name__ == '__main__':
    model = get_submean_model()
    m = np.random.random((3, 7))
    print(m)
    res = model.predict(m)
    print(type(res))
    print(res)
