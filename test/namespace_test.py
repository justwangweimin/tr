#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : namespace_test.py
# @Author: zjj421
# @Date  : 17-9-4
# @Desc  :

def get_var_name(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj][0]
def write_config(var):
    key = get_var_name(var)
    print(key)
    pyobj = {}
    pyobj[key] = var
    print(pyobj)
s1 = 2
s2 = "nihao"
s3 =[1, 2, 3]
# for i in [s1, s2, s3]:
for i in range(100):
    write_config(s1)
    write_config(s2)
    write_config(s3)
