#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : segfiles.py
# @Author: wangweimin
# @Date  : 17-11-8
# @Desc  :
import train.util as util
import train.const as const

"""
输入：src源文件夹，dst目标文件夹，num分成的分数
功能：将给定的源文件夹src中的文件，分成num份，放入dst文件夹中。
备注：dst文件夹中，会出现文件夹1,文件夹2,...,文件夹n。其中，n=num
"""
def run(srcpath, dstpath, num):
    # src中的文件数
    srcFullfilenames = util.getFullfilenamesFromPath(srcpath)
    srcFilenames=util.getFilenamesFromPath(srcpath)
    l=len(srcFullfilenames)
    j=1
    for i in range(l):
        src=srcFullfilenames[i]
        dst=dstpath.format(str(j))+"/"+srcFilenames[i]
        util.createSymlink(src, dst)
        j=j+1
        if j>num:
            j=1

if __name__ == '__main__':
    SRCDATAPATH = const.SRCDATAPATH
    DSTDATAPATH = const.DSTDATAPATH
    NUM = const.NUM
    run(SRCDATAPATH, DSTDATAPATH, NUM)
    print("All have done.")