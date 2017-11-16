#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : rw_conf_file.py
# @Author: zjj421
# @Date  : 17-9-3
# @Desc  :


import configparser

sectionname1 = 'imgdir'
sectionname2 = 'model'
dir_num = "V-{}".format(1)
model_num = "V-{}".format(1)
dir_name = "001"
model_name = "001"
conf = configparser.ConfigParser()
conf.add_section(sectionname1)
conf.set(sectionname1, dir_num, dir_name)
conf1 = configparser.ConfigParser()
conf1.add_section(sectionname2)
conf1.set(sectionname2, model_num, model_name)
with open('/home/zj/testforfun/conf.ini', 'w') as fw:
    conf.write(fw)
    conf1.write(fw)

# v1 = conf.getint(sectionname1, dir_num)
# v2 = conf.getint(sectionname1, dir_num)
# v3 = conf.getint(sectionname1, dir_num)
# v4 = conf.getint(sectionname1, dir_num)
# print('v1:', v1)
# print('v2:', v2)
# print('v3:', v3)
# print('v4:', v4)
