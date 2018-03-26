#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util.py
# @Author: wangweimin
# @Date  : 17-11-8
# @Desc  :
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import dice
import train.const as const

"""
input:path,one directory
return:all short file name in the path, not include ext and filepath
"""
def getShortfilenamesFromPath(path):
    filenames = []
    # root, dirs, files 分别表示：父目录名（全名），目录名，文件名
    # 如果dir_path不存在,循环内的语句不会被执行，而继续运行循环后面的语句
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.splitext(file)[0]
            filenames.append(filename)
    return filenames

"""
input:path,one directory
return:all file name in the path, not include filepath
"""
def getFilenamesFromPath(path):
    filenames = []
    # root, dirs, files 分别表示：父目录名（全名），目录名，文件名
    # 如果dir_path不存在,循环内的语句不会被执行，而继续运行循环后面的语句
    for root, dirs, files in os.walk(path):
        for file in files:
            filenames.append(file)
    return filenames

"""
input:path,one directory
return:all full file name in the path, not include ext and filepath
"""
def getFullfilenamesFromPath(path):
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames

"""
功能：遍历文件夹path，返回文件夹中的文件数
"""
def getFilenumFromPath(path):
    num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            num = num + 1
    return num

"""
输入：csv文件的路径，该csv文件包含两列：Id,Probability，其中Id由Subject和Zone组成。Subject是人的标识,Zone是人体区块的编号
"""
def getSubjectsFromCsv(csvfile):
    # 读入csv文件，成为df
    df = pd.read_csv(csvfile)
    # 将Id拆分成Subject和Zone
    df['Subject'], df['Zone'] = df['Id'].str.split('_', 1).str
    # 去除df中的Id列
    # df = df[['Subject', 'Zone', 'Probability']]
    subjects = list(df["Subject"])
    subjects = list(set(subjects))
    return subjects

"""
输入：csv文件的路径，该csv文件包含两列：Id,Probability，其中Id由Subject和Zone组成。Subject是人的标识,Zone是人体区块的编号
"""
def getLabelsdicFromCsv(csvfile):
    # 读入csv文件，成为df
    df = pd.read_csv(csvfile)
    # 将Id拆分成Subject和Zone
    df['Subject'], df['Zone'] = df['Id'].str.split('_Zone', 1).str
    # 去除df中的Id列
    # df = df[['Subject', 'Zone', 'Probability']]
    # return df
    dic={}
    l=len(df["Subject"])
    # print(l)
    for i in range(l):
        subject=df["Subject"][i]
        zone=df["Zone"][i]
        probability=df["Probability"][i]
        subject=str(subject)
        # print("Subject:{},Zone:{},Probability:{}".format(subject,zone,probability))
        idx=int(zone)-1
        keys=dic.keys()
        if subject in keys:
            dic[subject][idx]=probability
        else:
            dic[subject]=[0]*17
            dic[subject][idx]=probability
    return dic

"""
功能：创建一个指向src的符号链接dst，dst-->src
"""
def createSymlink(src, dst):
    if not os.path.exists(src):
        print("源文件不存在，无法创建软链接！")
        return
    if os.path.isfile(src):
        # 返回dst的目录（父目录全名）
        dirname = os.path.dirname(dst)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    os.symlink(src, dst)

"""
输入：datafilePath数据文件路径，labelcsv标签csv文件全名
功能：获取有效的Subjects
"""
def getValidSubjects(datafilePath, labelscsv):
    filenames = getShortfilenamesFromPath(datafilePath)
    subjects = getSubjectsFromCsv(labelscsv)
    aryFilenames = np.array(filenames)
    arySubjects = np.array(subjects)
    # 交集
    validSubjects = list(np.intersect1d(aryFilenames, arySubjects))
    return validSubjects


if __name__ == '__main__':
    LABELCSV = const.LABELCSV
    dic = getLabelsdicFromCsv(LABELCSV)
    print(dic)
