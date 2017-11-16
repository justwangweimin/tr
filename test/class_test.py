#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : class_test.py
# @Author: zjj421
# @Date  : 17-11-3
# @Desc  :


class BaseClass(object):
    def __init__(self, first_name=None, last_name=None, email=None):
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self._name = "hahha"
        print("构造函数已经运行。")

    @property
    def name(self):
        print("name属性已被读取！")
        # _name = self.first_name + self.last_name
        return self._name

    # @name.setter
    # def name(self, value=None):
    #     print("name属性已被设置!")
    #     self._name = value

    def modify_name(self):
        self._name = "1111111111"



class SubClass(BaseClass):
    def __init__(self, phone, *kwargs):
        super(SubClass, self).__init__(*kwargs)
        self.phone = phone


def __main():
    a = BaseClass("z", "j", "zj@163.com")
    print(a.name)
    # a.name = "zzzzzzzzzzz"
    # print(a.name)
    a.modify_name()
    print(a.name)

    # b = SubClass(110, "f", "ww", "fww@163.com")
    # print(b.name)
    # print(b.first_name)
    # print(b.last_name)
    # print(b.email)
    # print(b.phone)


if __name__ == '__main__':
    __main()
