#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : extractcnnfeature2h5.py
# @Author: wangweimin
# @Date  : 17-11-8
# @Desc  :
from datetime import datetime

import h5py
import numpy as np
from keras.applications import ResNet50, VGG16, VGG19, InceptionV3, Xception

import train.const as const
"""
功能：根据指定的特征名name，创建特征提取cnn模型
"""
def createCnn(name):
    if name == "VGG16":
        model = VGG16(weights="imagenet", include_top=False, pooling="avg")
    elif name == "VGG19":
        model = VGG19(weights="imagenet", include_top=False, pooling="avg")
    elif name == "ResNet50":
        model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    elif name == "InceptionV3":
        model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    elif name == "Xception":
        model = Xception(weights="imagenet", include_top=False, pooling="avg")
    # 意外报错　ValueError: When setting`include_top=True`, `input_shape` should be (224, 224, 3).
    # elif name == "MobileNet":
    #     model = MobileNet(weights="imagenet", include_top=False, input_shape=(200, 200, 3),
    #                       pooling="avg")
    else:
        model = None
        print("请输入正确的cnn预训练模型名")
        exit()
    # model.summary()
    return model

"""
功能：根据输入的特征名称name，用cnn得到图像特征，并保存
"""
def extractFeature(name, fImgs, fFeatures):
    model = createCnn(name)
    fImgsKeys = fImgs.keys()
    for i, key in enumerate(fImgsKeys):
        fFeaturesKeys=fFeatures.keys()
        # 保证该程序可中断
        if key in fFeaturesKeys:
            print("有重复的key")
            continue
        # print(key)
        imgs = fImgs[key].value
        features = []
        l=len(imgs)
        for j in range(l):
            img=imgs[j]
            shapelen = len(img.shape)
            if shapelen==2:
                img=[img for tmp in range(3)]
                img = np.array(img)
                img=np.transpose(img)
            img = np.expand_dims(img, axis=0)
            feature = model.predict_on_batch(img)
            feature = feature.reshape(-1)
            features.append(feature)
        features = np.asarray(features)
        fFeatures.create_dataset(str(key), data=features)
        # print("-" * 20, i + 1)

"""
功能：提取所有的cnn特征，并保存
"""
def extractFeatures():
    for i in range(NUM):
        k = i + 1
        # print("batch {}".format(k))
        fImgH5 = h5py.File(IMGH5FILENAME.format(str(k)), "a")
        for name in NAMES:
            fFeatureH5 = h5py.File(FEATUREH5FILENAME.format(name,str(k)), "a")
            extractFeature(name=name, fImgs=fImgH5, fFeatures=fFeatureH5)
            fFeatureH5.close()
        fImgH5.close()
        print(i+1,"-" * 30)

IMGH5FILENAME = const.IMGH5FILENAME
FEATUREH5FILENAME = const.FEATUREH5FILENAME
NAMES=const.NAMES
NUM=const.NUM

if __name__ == '__main__':
    begin = datetime.now()
    print("开始时间： ", begin)
    extractFeatures()
    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))