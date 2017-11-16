#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : np_fromfile_test.py
# @Author: zjj421
# @Date  : 17-8-27
# @Desc  :
# import numpy as np
# a = [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
# a.tofile('a.txt')
# b = np.fromfile('a.txt', dtype=np.int32)
# print(b)

import numpy as np

dt = np.dtype([('time', [('min', int), ('sec', int)]), ('temp', float)])
x = np.zeros((1,), dtype=dt)
x['time']['min'] = 10
x['temp'] = 98.25
print(x)
