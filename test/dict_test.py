#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : dict_test.py
# @Author: zjj421
# @Date  : 17-9-8
# @Desc  :

dct = {"name": "zj", "age": 18}


def f():
    try:
        s = dct["gender"]
    except:
        return

ff = f()
print(ff)
