#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : const.py
# @Author: wangweimin
# @Date  : 17-11-8
# @Desc  :
from keras import optimizers
#版本0,2:图像处理成灰度图，进行了缩放，然后变成彩色图保存
#版本1：图像处理成灰度图，不进行缩放，然后变成彩色图保存
VERSION=1
STAGE="1"
#数据文件路径
SRCDATAPATH = "../../input/stage"+STAGE+"/stage1_aps"
#分割后的文件夹个数
NUM=10
#分开后的目地数据文件路径：数据文件路径下的数据文件太多，不能一次性处理，需要分开。
DSTDATAPATH = "../../input/stage"+STAGE+"/stage1_aps_dst/{}"
#图像h5文件名
IMGH5FILENAME = "../../output/stage"+STAGE+"/h5v"+str(VERSION)+"/trainimg{}.h5"
#特征h5文件名
FEATUREH5FILENAME = "../../output/stage"+STAGE+"/h5v"+str(VERSION)+"/trainfeature{}_{}.h5"
#特征名数组
NAMES=["InceptionV3","VGG16","VGG19","ResNet50","Xception"]
DS={"InceptionV3":2048,"VGG16":512,"VGG19":512,"ResNet50":2048,"Xception":2048}
#采用的图像特征
NAME="VGG16"
LABELCSV="../../input/stage"+STAGE+"/stage1_labels.csv"
TESTLABELCSV="../../input/stage"+STAGE+"/test_labels.csv"
INPUTSHAPE=(DS[NAME],)

#SIZE=299
SIZE=200
EPOCS=300
BATCH_SIZE=64
PREDICT_BATCH_SIZE=10
EVALUATE_BATCH_SIZE=10
FLAG=3
GD= optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#缩放的图像大小
TARGETIMGSIZE = (SIZE, SIZE)

MODELFILENAME=("../../output/stage"+STAGE+"/model/m_{}_{}.h5").format(NAME,str(FLAG-1))
if FLAG>=10:
    FLAG=1
NEWMODELFILENAME=("../../output/stage"+STAGE+"/model/m_{}_{}.h5").format(NAME,str(FLAG))
FITRESULTCSV=("../../output/stage"+STAGE+"/model/fit_{}_{}.csv").format(NAME,str(FLAG))
SUBMISSIONCSV=("../../output/stage"+STAGE+"/model/submit_{}_{}.csv").format(NAME,str(FLAG))