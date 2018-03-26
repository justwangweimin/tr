#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : createmodel.py
# @Author: wangweimin
# @Date  : 17-11-8
# @Desc  :
from keras import Input, models,optimizers
from keras.applications import ResNet50, VGG16, VGG19, InceptionV3, Xception
from keras.layers import Dense, TimeDistributed, LSTM, concatenate, Lambda, Dropout
import train.const as const


def createCnn(name):
    if name == "VGG16":
        cnn = VGG16(weights="imagenet", include_top=False, pooling="avg")
    elif name == "VGG19":
        cnn = VGG19(weights="imagenet", include_top=False, pooling="avg")
    elif name == "ResNet50":
        cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    elif name == "InceptionV3":
        cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    elif name == "Xception":
        cnn = Xception(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False  # 参数不可训练（改变）
    return cnn
"""
输入：inputshape=(150, 150, 3)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理图像
备注：输入的是1个图像
"""
def createModelPrecessImg(inputshape,name):
    # Input层
    #pic = Input(shape=(150, 150, 3), name="pic")
    pic = Input(shape=inputshape, name="pic")
    # cnn层， avg池化
    cnn=createCnn(name)
    # 一个cnn处理一张图片
    frame_features = cnn(pic)
    output_voc_size = 17
    outputs = Dense(output_voc_size, name="predictions", activation="sigmoid")(frame_features)
    model = models.Model(inputs=pic, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(None, 150, 150, 3)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个图像。使用LSTM
备注：输入的是16个图像，1个输入
"""
def createModelProcessImgsWithLSTM(inputshape,name):
    # Input层
    pics = Input(shape=inputshape, name="pics")
    # cnn层， avg池化
    cnn = createCnn(name)
    frame_features = TimeDistributed(cnn)(pics)
    pics_vector = LSTM(256)(frame_features)
    output_voc_size = 17
    outputs = Dense(output_voc_size, name="predictions", activation="sigmoid")(pics_vector)
    model = models.Model(inputs=pics, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(150, 150, 3)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个图像。不使用LSTM
备注：输入的是16个图像，1个图像一个输入，共16个输入。
(16个图像)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessImgsWithoutLSTM1(inputshape,name):
    # Input层
    inputs = []
    frame_features = []
    # cnn层， avg池化
    cnn = createCnn(name)
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        frame_feature = cnn(input)
        inputs.append(input)
        frame_features.append(frame_feature)
    frame_features = concatenate(frame_features, axis=-1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(150, 150, 3)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个图像。不使用LSTM
备注：输入的是16个图像，1个图像一个输入，共16个输入。
(16个图像)-->(分别提取特征，得到16个图像特征)-->(4个一组，降维，形成4个特征)...
"""
def createModelProcessImgsWithoutLSTM2(inputshape,name):
    # Input层
    inputs = []
    frame_features = []
    # cnn层， avg池化
    cnn = createCnn(name)
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        frame_feature = cnn(input)
        inputs.append(input)
        frame_features.append(frame_feature)
    frame_features_1 = concatenate(frame_features[0:4], axis=-1)
    frame_features_2 = concatenate(frame_features[4:8], axis=-1)
    frame_features_3 = concatenate(frame_features[8:12], axis=-1)
    frame_features_4 = concatenate(frame_features[12:16], axis=-1)
    y1 = Dense(2048, activation="tanh")(frame_features_1)
    y2 = Dense(2048, activation="tanh")(frame_features_2)
    y3 = Dense(2048, activation="tanh")(frame_features_3)
    y4 = Dense(2048, activation="tanh")(frame_features_4)
    frame_features = [y1, y2, y3, y4]
    frame_features = concatenate(frame_features, axis=-1)

    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    y = Dense(2048, activation="tanh")(frame_features)
    y = Dense(256, activation="tanh")(y)

    output_voc_size = 17
    outputs = Dense(output_voc_size, activation="sigmoid")(y)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(150, 150, 3)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个图像。不使用LSTM
备注：输入的是16个图像，1个图像一个输入，共16个输入。
(16个图像)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessImgsWithoutLSTM3(inputshape,name):
    # Input层
    inputs = []
    frame_features = []
    # cnn层， avg池化
    cnn = createCnn(name)
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        frame_feature = cnn(input)
        inputs.append(input)
        frame_features.append(frame_feature)
    frame_features = concatenate(frame_features, axis=-1)
    frame_features = Dense(2048, activation="tanh")(frame_features)
    frame_features = Dense(256, activation="tanh")(frame_features)
    frame_features = Dense(64, activation="tanh")(frame_features)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(150, 150, 3)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个图像。不使用LSTM
备注：输入的是16个图像，1个图像一个输入，共16个输入。
(16个图像)-->(分别提取特征，得到16个图像特征)-->(4个一组，降维，形成4个特征)...
"""
def createModelProcessImgsWithoutLSTM4(inputshape,name):
    # Input层
    inputs = []
    frame_features = []
    # cnn层， avg池化
    cnn = createCnn(name)
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        frame_feature = cnn(input)
        inputs.append(input)
        frame_features.append(frame_feature)
    frame_features_1 = concatenate(frame_features[0:4], axis=-1)
    frame_features_2 = concatenate(frame_features[4:8], axis=-1)
    frame_features_3 = concatenate(frame_features[8:12], axis=-1)
    frame_features_4 = concatenate(frame_features[12:16], axis=-1)
    y1 = Dense(2048, activation="tanh")(frame_features_1)
    y2 = Dense(2048, activation="tanh")(frame_features_2)
    y3 = Dense(2048, activation="tanh")(frame_features_3)
    y4 = Dense(2048, activation="tanh")(frame_features_4)
    frame_features = [y1, y2, y3, y4]
    frame_features = concatenate(frame_features, axis=-1)

    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    y = Dense(2048, activation="tanh")(frame_features)
    y = Dense(512, activation="tanh")(y)
    y = Dense(128, activation="tanh")(y)

    output_voc_size = 17
    outputs = Dense(output_voc_size, activation="sigmoid")(y)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(150, 150, 3)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个图像。不使用LSTM
备注：输入的是16个图像，1个图像一个输入，共16个输入。
(16个图像)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessImgsWithoutLSTM5(inputshape,name):
    # Input层
    inputs = []
    frame_features = []
    # cnn层， avg池化
    cnn = createCnn(name)
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        frame_feature = cnn(input)
        inputs.append(input)
        frame_features.append(frame_feature)
    frame_features = concatenate(frame_features, axis=-1)
    frame_features = Dense(2048, activation="relu")(frame_features)
    frame_features = Dense(256, activation="relu")(frame_features)
    frame_features = Dense(64, activation="relu")(frame_features)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    y = Dense(output_voc_size, activation="sigmoid")(frame_features)
    model = models.Model(inputs=inputs, outputs=y)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(2048,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM(inputshape):
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(2048, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(2048,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM2(inputshape):
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(2048 * 2, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(1024, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(2048,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM3(inputshape):
    inputs = []
    num_angles = 8
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(2048 * 2, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(1024, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM4(inputshape):
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(512 * 2, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM5(inputshape):
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dropout(rate=0.2)(frame_features_1)
    frame_features_1 = Dense(512 * 4, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="relu")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    # sgd = Nadam()
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# 模型1002的改进
"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM6(inputshape):
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(512 * 2, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="relu")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="binary_crossentropy")
    return model


# 模型1002的改进
"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM7(inputshape):
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(512 * 2, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="binary_crossentropy")
    return model


# 模型1002的改进：增加参数个数。
"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM8(inputshape):
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(512 * 4, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(512, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(128, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# 模型1002的改进：继续增加参数个数。
"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM9(inputshape):
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(512 * 8, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(512 * 4, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(512 * 2, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(512, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(128, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(32, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM10(inputshape):
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dropout(rate=0.2)(frame_features_1)
    frame_features_1 = Dense(512 * 8, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(512 * 4, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(512 * 2, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(512, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(128, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(32, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    gd = const.GD
    model.compile(optimizer=gd, loss="binary_crossentropy")
    return model

# 模型1002的改进：继续增加参数个数。
# 该问题激活函数不能用softmax,训练效果极差.
"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM2_BAD1(inputshape):
    inputs = []
    num_angles = 8
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(512 * 4, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(512 * 2, activation="relu")(frame_features_1)
    frame_features_1 = Dense(512, activation="tanh")(frame_features_1)
    frame_features_1 = Dropout(rate=0.2)(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(128, activation="relu")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(32, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="softmax")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model

# 模型1002的改进：继续增加参数个数。
# 该问题激活函数不能用softmax,训练效果极差.
"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithoutLSTM2_BAD2(inputshape):
    inputs = []
    num_angles = 8
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(512 * 2, activation="relu")(frame_features_1)
    frame_features_1 = Dropout(rate=0.3)(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# LSTM模型,输入特征shape为(512,)
"""
输入：inputshape=(None,512)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithLSTM1(inputshape):
    # Input层
    inputs = Input(shape=inputshape, name="pics")
    frame_features = TimeDistributed(Lambda(lambda x: x))(inputs)
    # frame_features = TimeDistributed(Dense(16))(inputs)
    pics_vector = LSTM(256)(frame_features)
    pics_vector = Dense(64, activation="tanh")(pics_vector)
    output_voc_size = 17
    outputs = Dense(output_voc_size, name="predictions", activation="sigmoid")(pics_vector)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# LSTM模型,输入特征shape为(512,), 由3001改动得到.
"""
输入：inputshape=(None,512)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithLSTM2(inputshape):
    # Input层
    inputs = Input(shape=inputshape, name="pics")
    frame_features = TimeDistributed(Lambda(lambda x: x))(inputs)
    # frame_features = TimeDistributed(Dense(16))(inputs)
    pics_vector = LSTM(512)(frame_features)
    pics_vector = Dense(256, activation="tanh")(pics_vector)
    pics_vector = Dense(64, activation="tanh")(pics_vector)
    output_voc_size = 17
    outputs = Dense(output_voc_size, name="predictions", activation="sigmoid")(pics_vector)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


"""
输入：inputshape=(None,2048)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeaturesWithLSTM3(inputshape):
    # Input层
    inputs = Input(shape=inputshape, name="pics")
    frame_features = TimeDistributed(Lambda(lambda x: x))(inputs)
    # frame_features = TimeDistributed(Dense(16))(inputs)
    pics_vector = LSTM(2048)(frame_features)
    pics_vector = Dense(256, activation="tanh")(pics_vector)
    pics_vector = Dense(64, activation="tanh")(pics_vector)
    output_voc_size = 17
    outputs = Dense(output_voc_size, name="predictions", activation="sigmoid")(pics_vector)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# 64个输入,单个输入shape为(512,)。
"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeatures64WithoutLSTM1(inputshape):
    inputs = []
    num_angles = 64
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(512 * 2, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# 64个输入,单个输入shape为(512,)。
"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeatures64WithoutLSTM2(inputshape):
    inputs = []
    num_angles = 64
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    batch = 0
    frame_features_1 = []
    for i in range(num_angles // 4):
        frame_features_ = concatenate(inputs[batch:batch + 4])
        batch = batch + 4
        frame_features_ = Dense(512, activation="tanh")(frame_features_)
        frame_features_1.append(frame_features_)
    frame_features_2 = []
    batch = 0
    for i in range(num_angles // 4 // 4):
        frame_features_ = concatenate(frame_features_1[batch: batch + 4])
        batch = batch + 4
        frame_features_ = Dense(512, activation="tanh")(frame_features_)
        frame_features_2.append(frame_features_)
    frame_features_3 = concatenate(frame_features_2)
    frame_features_4 = Dense(512, activation="tanh")(frame_features_3)

    frame_features_4 = Dense(256, activation="tanh")(frame_features_4)
    frame_features_4 = Dense(64, activation="tanh")(frame_features_4)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_4)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# 64个输入,单个输入shape为(512,)。
"""
输入：inputshape=(512,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeatures64WithoutLSTM3(inputshape):
    inputs = []
    num_angles = 64
    for i in range(num_angles):
        input = Input(shape=inputshape)
        inputs.append(input)
    frame_features_1 = concatenate(inputs)
    frame_features_1 = Dense(512 * 3, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(512, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="relu")(frame_features_1)
    frame_features_1 = Dense(64, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# 新模型，特征融合
"""
输入：inputshape=(2048,)，name属于{VGG16,VGG19,ResNet50,InceptionV3,Xception}
功能：创建模型，用于处理16个特征。不使用LSTM
备注：输入的是16个特征，1个特征一个输入，共16个输入。
(16个特征)-->(分别提取特征，得到16个图像特征)-->(16个一组，降维，形成1个特征)...
"""
def createModelProcessFeatures64WithoutLSTM4(inputshape):
    # Input层
    inputs = []
    num_angles = 16
    num_cnns = 2
    for i in range(num_cnns):
        for j in range(num_angles):
            input = Input(shape=inputshape)
            inputs.append(input)
    frame_features_1 = concatenate(inputs[0:num_angles])
    frame_features_1 = Dense(2048, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(256, activation="tanh")(frame_features_1)
    frame_features_2 = concatenate(inputs[num_angles:num_angles * 2])
    frame_features_2 = Dense(2048, activation="tanh")(frame_features_2)
    frame_features_2 = Dense(256, activation="tanh")(frame_features_2)
    frame_features = concatenate([frame_features_1, frame_features_2])
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model