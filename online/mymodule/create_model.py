#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : create_model.py
# @Author: zjj421
# @Date  : 17-9-11
# @Desc  :
from keras import Input, models
from keras.applications import InceptionV3
from keras.layers import Dense, TimeDistributed, LSTM, concatenate, Reshape, Lambda


def create_model_1():
    # Input层
    pic = Input(shape=(150, 150, 3), name="pic")
    # InceptionV3层， avg池化
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False  # 参数不可训练（改变）
    # 一个cnn处理一张图片
    frame_features = cnn(pic)
    output_voc_size = 17
    outputs = Dense(output_voc_size, name="predictions", activation="sigmoid")(frame_features)
    model = models.Model(inputs=pic, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


def create_model_2():
    # Input层
    pics = Input(shape=(None, 150, 150, 3), name="pics")
    # InceptionV3层， avg池化
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False  # 参数不可训练（改变）
    frame_features = TimeDistributed(cnn)(pics)
    pics_vector = LSTM(256)(frame_features)
    output_voc_size = 17
    outputs = Dense(output_voc_size, name="predictions", activation="sigmoid")(pics_vector)
    model = models.Model(inputs=pics, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


def create_model_3():
    # Input层
    inputs = []
    frame_features = []
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False  # 参数不可训练（改变）
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(150, 150, 3))
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


def create_model_4():
    # Input层
    inputs = []
    frame_features = []
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False  # 参数不可训练（改变）
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(150, 150, 3))
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


def create_model_5():
    # Input层
    inputs = []
    frame_features = []
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False  # 参数不可训练（改变）
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(150, 150, 3))
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


def create_model_6():
    # Input层
    inputs = []
    frame_features = []
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False  # 参数不可训练（改变）
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(150, 150, 3))
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


def create_model_7():
    # Input层
    inputs = []
    frame_features = []
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False  # 参数不可训练（改变）
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(150, 150, 3))
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


def create_model_1001():
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(2048,))
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


def create_model_1002():
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(512,))
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


# 模型1002的改进
def create_model_1003():
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(512,))
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
def create_model_1004():
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(512,))
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
def create_model_1005():
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(512,))
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
def create_model_1006():
    inputs = []
    num_angles = 16
    for i in range(num_angles):
        input = Input(shape=(512,))
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


# LSTM模型,输入特征shape为(512,)
def create_model_3001():
    # Input层
    inputs = Input(shape=(None, 512), name="pics")
    frame_features = TimeDistributed(Lambda(lambda x:x))(inputs)
    # frame_features = TimeDistributed(Dense(16))(inputs)
    pics_vector = LSTM(256)(frame_features)
    pics_vector = Dense(64, activation="tanh")(pics_vector)
    output_voc_size = 17
    outputs = Dense(output_voc_size, name="predictions", activation="sigmoid")(pics_vector)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


# LSTM模型,输入特征shape为(2048,)
def create_model_3002():
    # Input层
    inputs = Input(shape=(None, 2048), name="pics")
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


# 新模型，特征融合
def create_model_2001():
    # Input层
    inputs = []
    num_angles = 16
    num_cnns = 2
    for i in range(num_cnns):
        for j in range(num_angles):
            input = Input(shape=(2048,))
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


def create_model_9001():
    inputs = Input(shape=(17,))
    frame_features_1 = Dense(17, activation="tanh")(inputs)
    frame_features_1 = Dense(17, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(17, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(17, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(17, activation="tanh")(frame_features_1)
    frame_features_1 = Dense(17, activation="tanh")(frame_features_1)
    output_voc_size = 17
    # 全链接层，输入维度太高，会导致数据稀疏，使得loss很难收敛。
    outputs = Dense(output_voc_size, activation="sigmoid")(frame_features_1)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    return model


def create_model(which_model):
    if which_model == 1:
        model = create_model_1()
    elif which_model == 2:
        model = create_model_2()
    elif which_model == 3:
        model = create_model_3()
    elif which_model == 4:
        model = create_model_4()
    elif which_model == 5:
        model = create_model_5()
    elif which_model == 6:
        model = create_model_6()
    elif which_model == 7:
        model = create_model_7()
    elif which_model == 1001:
        model = create_model_1001()
    elif which_model == 1002:
        model = create_model_1002()
    elif which_model == 1003:
        model = create_model_1003()
    elif which_model == 1004:
        model = create_model_1004()
    elif which_model == 1005:
        model = create_model_1005()
    elif which_model == 1006:
        model = create_model_1006()
    elif which_model == 2001:
        model = create_model_2001()
    elif which_model == 3001:
        model = create_model_3001()
    elif which_model == 3002:
        model = create_model_3002()
    elif which_model == 9001:
        model = create_model_9001()
    else:
        model = None
        print("请设置好WHICH_MODEL!")
        exit()
    # 打印模型
    model.summary()
    print("模型 {} 创建完毕！".format(which_model))
    print("-" * 100)
    return model
