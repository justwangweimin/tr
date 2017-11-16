#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : varname.py
# @Author: zjj421
# @Date  : 17-9-8
# @Desc  :

import inspect


def get_var_name(var):
    '''
    utils:
    get back the name of variables
    '''
    # callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]




def f2(fp):
    f4 = get_var_name(fp)
    print(f4, fp)
f3 = "hihi"
f2(f3)

