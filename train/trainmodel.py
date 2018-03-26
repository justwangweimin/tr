#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : trainmodel.py
# @Author: wangweimin
# @Date  : 17-11-8
# @Desc  :
import h5py
import os
import pandas as pd
import numpy as np
from keras.models import save_model, load_model
from datetime import datetime

import train.const as const
import train.util as util
import train.createmodel as createmodel
import train.computescore as computescore

"""
功能：保存模型的fit结果
"""
def saveFitResult(hist, fitresultcsv):
    df = pd.DataFrame()
    if "loss" in hist:
        df["loss"] = hist["loss"]
    if "val_loss" in hist:
        df["val_loss"] = hist["val_loss"]
    df.to_csv(fitresultcsv, index=False)
    print("*" * 100)
    print('输出结果 "{}" 已保存！'.format(fitresultcsv))

"""
功能：生成模型
"""
def generateModel():
    if os.path.exists(MODELFILENAME):
        model = load_model(MODELFILENAME)
        print("模型权重'{}'导入成功".format(MODELFILENAME))
    else:
        model = createmodel.createModelProcessFeaturesWithoutLSTM10(INPUTSHAPE)
    return model

"""
功能：训练模型
"""
def trainModel(model, train_x, train_y, val_x=None, val_y=None):
    if val_x is None and val_y is None:
        validation_data = None
        validation_split=0
    else:
        validation_data = (val_x, val_y)
        validation_split = len(val_x) / len(train_y)
    hist = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                     validation_data=validation_data, validation_split=validation_split)
    saveFitResult(hist.history, FITRESULTCSV)
    save_model(model, NEWMODELFILENAME, overwrite=True, include_optimizer=True)

"""
功能：准备训练和测试数据
"""
def prepareData():
    dic = util.getLabelsdicFromCsv(LABELCSV)
    ret = {}
    ret["train_x"] = [[] for i in range(16)]
    ret["train_y"] = []
    ret["train_subjects"] = []
    ret["test_x"] = [[] for i in range(16)]
    ret["test_subjects"] = []
    for i in range(NUM):
        k = i + 1
        featurefilename = FEATUREH5FILENAME.format(NAME, str(k))
        fFeatureH5 = h5py.File(featurefilename, "r")
        # 特征h5文件的key是subject，value是16个图像特征
        keys = fFeatureH5.keys()
        for subject in keys:
            features = fFeatureH5[subject]
            if subject in dic.keys():
                ret["train_y"].append(dic[subject])
                ret["train_subjects"].append(subject)
                for j in range(16):
                    ret["train_x"][j].append(features[j])
            else:
                ret["test_subjects"].append(subject)
                for j in range(16):
                    ret["test_x"][j].append(features[j])
        fFeatureH5.close()
    return ret

"""
功能：评估模型
"""
def evaluateModel(model, val_x, val_y):
    val_loss = model.evaluate(val_x, val_y, EVALUATE_BATCH_SIZE)
    print("\nval_loss: {}".format(int(10 ** 5 * val_loss) / (10 ** 5)))

"""
功能：预测模型
"""
def predictModel(model, test_x, test_subjects):
    predict_values = model.predict(test_x, batch_size=PREDICT_BATCH_SIZE, verbose=1)
    print("\nmodel.predict的输出shape: ", predict_values.shape)

    l=len(test_subjects)
    t=[[0 for j in range(17)] for i in range(l)]
    for i in range(l):
        subject=test_subjects[i]
        for j in range(17):
            t[i][j]=subject+"_Zone"+str(j+1)
    t=np.array(t)
    t=t.reshape(-1)
    predict_values = predict_values.reshape(-1)

    dataframe = pd.DataFrame({"Id": t, "Probability": predict_values})
    dataframe.to_csv(SUBMISSIONCSV, index=False, sep=",")
    print('文件"{}"已保存！'.format(SUBMISSIONCSV))

if __name__ == '__main__':
    LABELCSV = const.LABELCSV
    FEATUREH5FILENAME = const.FEATUREH5FILENAME
    NAMES = const.NAMES
    NUM = const.NUM
    NAME = const.NAME
    INPUTSHAPE = const.INPUTSHAPE
    MODELFILENAME = const.MODELFILENAME
    NEWMODELFILENAME=const.NEWMODELFILENAME
    BATCH_SIZE = const.BATCH_SIZE
    EPOCHS = const.EPOCS
    FITRESULTCSV = const.FITRESULTCSV
    PREDICT_BATCH_SIZE = const.PREDICT_BATCH_SIZE
    EVALUATE_BATCH_SIZE = const.EVALUATE_BATCH_SIZE
    SUBMISSIONCSV = const.SUBMISSIONCSV
    GD=const.GD

    begin = datetime.now()
    print("开始时间： ", begin)
    data = prepareData()

    model = generateModel()

    # 打印模型
    model.summary()
    # exit()

    train_x = list(np.array(data["train_x"]))
    train_y = data["train_y"]
    val_x = None
    val_y = None
    trainModel(model, train_x, train_y, val_x, val_y)
    test_x = list(np.array(data["test_x"]))
    test_subjects = data["test_subjects"]
    predictModel(model, test_x, test_subjects)

    computescore.run()
    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))