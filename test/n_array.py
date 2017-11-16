#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : n_array.py
# @Author: zjj421
# @Date  : 17-9-16
# @Desc  :
import numpy as np

arr = np.arange(1, 10 * 6 * 7 + 1).reshape(-1, 7)
# print(arr)

arr1 = arr[0:6]
print(arr1)
arr_1 = []
for i1, i2, i3, i4, i5, i6 in zip(arr1[0], arr1[1], arr1[2], arr1[3], arr1[4], arr1[5]):
    sum = i1 + i2 + i3 + i4 + i5 + i6
    m = sum / 6
    arr_1.append(m)
print(arr_1)
