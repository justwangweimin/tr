#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : custom_activation.py
# @Author: zjj421
# @Date  : 17-9-28
# @Desc  :

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def my_binary_step(x):
    if x < 0:
        x = 10 ** (-15)
    else:
        x = 1 - 10 ** (-15)
    return x


def d_my_binary_step(x):  # derivative
    if x < 0:
        x = 10 ** (-15)
    else:
        x = 1 - 10 ** (-15)
    return x


np_my_binary_step = np.vectorize(my_binary_step)

np_my_binary_step_32 = lambda x: np_my_binary_step(x).astype(np.float32)

np_d_my_binary_step = np.vectorize(d_my_binary_step)

np_d_my_binary_step_32 = lambda x: np_d_my_binary_step(x).astype(np.float32)


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_d_my_binary_step(x, name=None):
    with ops.op_scope([x], name, "d_my_binary_step") as name:
        y = tf.py_func(np_d_my_binary_step_32,
                       [x],
                       [tf.float32],
                       name=name,
                       stateful=False)
        return y[0]


def my_binary_step_grad(op, grad):
    x = op.inputs[0]

    n_gr = tf_d_my_binary_step(x)
    return grad * n_gr


def tf_my_binary_step(x, name=None):
    with ops.op_scope([x], name, "my_binary_step") as name:
        y = py_func(np_my_binary_step_32,
                    [x],
                    [tf.float32],
                    name=name,
                    grad=my_binary_step_grad)  # <-- here's the call to the gradient
        return y[0]
