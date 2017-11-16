#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : multiply_array.py
# @Author: zjj421
# @Date  : 17-9-17
# @Desc  :
import numpy as np

a = np.arange(6).reshape(2, 3)

print(a)
b = np.arange(1, 7).reshape(2, 3)
print(b)
print(a * b)
