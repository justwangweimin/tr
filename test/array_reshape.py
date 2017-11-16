#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : array_reshape.py
# @Author: zjj421
# @Date  : 17-9-8
# @Desc  :
import numpy as np
#
# arr = np.arange(10 * 9 * 8 * 7)
# arr = arr.reshape(10, 9, 8, 7)
# print(arr.shape)
# print(arr)
# arr = arr.reshape(90, 8, 7)
#
# print(arr.shape)
# print(arr)


a = np.arange(10)
print(a)

aa = [x for x in a for i in range(3)]
print(aa)