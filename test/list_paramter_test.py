#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:zhangjian 
@file: list_paramter_test.py 
@time: 2017/08/30 
"""
import numpy as np
a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]
x = [a, b, c]
print(x)
c1 = x.count(1)
print(c1)
# print(type(x))
# print(len(x))
# for i, s in enumerate(x):
#     print(i, s)
# x = np.array(x)
# y = x.reshape(x.size)
# print(type(x))
# print(y)




t = (1, 2, 3, 4)
for i, v in enumerate(t, 1):
    print(i, v)


