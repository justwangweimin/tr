#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : multiple_Inheritance.py
# @Author: zjj421
# @Date  : 17-11-9
# @Desc  :

class A:
    def __init__(self):
        print("A构造方法已执行.")

class B:
    def __init__(self):
        # A.__init__(self)
        # super().__init__()
        print("B构造方法已执行.")

class C(B, A):
    def __init__(self):
        # super(C, self).__init__()
        A.__init__(self)
        B.__init__(self)
        print("C构造方法已执行.")

def __main():
    c = C()


if __name__ == '__main__':
    __main()