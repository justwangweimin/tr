#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : preprocess_data.py
# @Author: wangweimin
# @Date  : 17-11-8
# @Desc  :
import os

import cv2
import numpy as np
import h5py
from tsahelper.tsahelper import get_subject_labels, get_subject_zone_label, read_data, convert_to_grayscale
from datetime import datetime

import train.util as util
import train.const as const
"""
功能：读入一个数据文件file，变成多个图
备注：1个数据file中包含16张图，读入后的格式是512,660,16
"""
def readData(file):
    print(file)
    imgs1 = read_data(file)
    # print("原始16个不同视图的图片的shape: {}".format(imgs.shape))
    # ----------------------------------------------------
    # 图片的shape（高，宽）与（宽，高）对图片本身来说是否一样？？
    # (512,660,16) --> (16, 660, 512)
    imgs1 = imgs1.transpose()
    # print("转换过的不同视图的图片的shape: {}".format(imgs.shape))
    return imgs1

"""
功能：给定的数据文件路径path，获取该路径中的所有的数据文件的数据,其中，每个数据文件将获取它的图像内容和Subject
备注：数据文件的文件名=Subject，其文件内容是16张图
     16个图  subject1
     16个图  subject2
     16个图  subject3
     ......  ........
"""
def getLstImgsAndSubjects(path):
    fullnames=util.getFullfilenamesFromPath(path)
    shortnames=util.getShortfilenamesFromPath(path)
    l=len(fullnames)
    lstImgs = []
    # 每个subject分为17个区块，id由subject_id+区块名组成
    subject_id_set = []
    for i in range(l):
        fullname=fullnames[i]
        imgs = readData(fullname)
        lstImgs.append(imgs)
    return lstImgs, shortnames

"""
功能：图像预处理，返回处理后的新图.----弃用
"""
def preprocessImg(img):
    newImg=img
    # correct the orientation of the image==纠正图像宽高==图像的上下翻转是不必要的，故注销
    # newImg = np.flipud(newImg) #上下反转，在本程序中无用
    # 转灰度图
    newImg = convert_to_grayscale(newImg)
    # 图片缩放成
    if VERSION != 1:
        newImg = cv2.resize(newImg, TARGETIMGSIZE, interpolation=cv2.INTER_AREA)  # return a new picture
    # 灰度图片转RGB
    newImg = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    return newImg

"""
功能：处理图像数组列表
"""
def preprocessLstImgs(lstImgs):
    # 处理后的所有图片数据
    newLstImgs= []
    # 图片resize和gray2RGB
    for imgs in lstImgs:
        newImgs = []
        for img in imgs:
            newImg=preprocessImg(img)
            newImgs.append(newImg)
        newLstImgs.append(newImgs)
    return newLstImgs


def zipData(x, y):
    z = []
    for pair in zip(x, y):
        z.append(pair)
    return z

"""
输入：path是要处理的数据文件夹，filename是保存的h5文件全名
功能：对数据文件夹中的所有数据文件进行预处理，结果存入1个h5文件
"""
def save2H5(path,f):
    lstImgs,subjects=getLstImgsAndSubjects(path)
    lstImgs=preprocessLstImgs(lstImgs)
    l= len(lstImgs)
    print(l)
    for i in range(l):
        imgs=lstImgs[i]
        subject=str(subjects[i])
        keys=f.keys()
        if subject in keys:
            continue
        f.create_dataset(subject,data=imgs, compression="gzip", compression_opts=4)

def run():
    for i in range(NUM):
        k=i+1
        print("batch {}".format(k))
        f = h5py.File(IMGH5FILENAME.format(str(k)), "a")
        save2H5(DATAPATH.format(str(k)),f)
        f.close()

TARGETIMGSIZE = const.TARGETIMGSIZE
DATAPATH = const.DSTDATAPATH
IMGH5FILENAME=const.IMGH5FILENAME
NUM=const.NUM
VERSION=const.VERSION
if __name__ == '__main__':
    begin = datetime.now()
    print("开始时间： ", begin)
    run()
    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))