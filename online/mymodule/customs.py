#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : customs.py
# @Author: zjj421
# @Date  : 17-9-27
# @Desc  :

from keras.engine import Layer
from keras.layers import initializers, activations


class MySubtractionLayer(Layer):
    def __init__(self, units=17, kernel_initializer='zeros', activation=None, **kwargs):
        super(MySubtractionLayer, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)

    def build(self, input_shape):
        # assert input_shape == (17,)
        print("自定义层input_shape: ", input_shape)
        assert input_shape[-1] == self.units
        self.kernel = self.add_weight(shape=(self.units,),
                                      initializer=self.kernel_initializer,
                                      name='kernel')
        self.build = True

    def call(self, inputs, **kwargs):
        print("inputs.shape:", inputs.shape)
        print("type(inputs):", type(inputs))
        print("type(self.kernel):", type(self.kernel))
        output = inputs - self.kernel
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
