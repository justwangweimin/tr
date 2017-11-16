#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : caculate.py
# @Author: zjj421
# @Date  : 17-9-26
# @Desc  :
# from numpy import mean
#
# s = [0.86228622, 0.44440924, 0.24732536, 0.5028154, 0.16541419, 0.32702348,
#      0.48084274]
# print(mean(s))
# [-0.82440448 -0.10729417  0.78016996  0.86862761 -0.71709889]

import numpy as np
a = np.array([[1, 1, 1],
     [2, 2, 2],
     [3, 3, 3]])
b = np.array([1, 1, 1])
print(a-b)
