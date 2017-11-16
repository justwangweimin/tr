#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : sub_list_test.py
# @Author: zjj421
# @Date  : 17-10-17
# @Desc  :
import numpy as np
from pandas import Series

def __main():
    l1 =np.array([1, 2, 3])
    l2 = Series([5, 5, 5])
    l3 = l1-l2
    print(type(l3))
    a = abs(l3)
    print(a)
    s = sum(a)
    print(s)

if __name__ == '__main__':
    __main()