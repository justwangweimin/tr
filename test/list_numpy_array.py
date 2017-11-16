#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : list_numpy_array.py
# @Author: zjj421
# @Date  : 17-8-31
# @Desc  :

import numpy as np

a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]
x = [a, b, c]
x = np.array(x)
y = []
y.append(x)
print(y)