#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : extract_cnn_features.py
# @Author: zjj421
# @Date  : 17-10-8
# @Desc  :
from datetime import datetime

import h5py
import numpy as np
from keras.applications import ResNet50, VGG16, VGG19, InceptionV3, Xception


def create_cnn_model(name):
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


def extract_cnn_features(cnn_name, f_imgs, f_features):
    print(cnn_name)
    model = create_cnn_model(cnn_name)
    f_imgs = f_imgs["imgs_trained"]
    # f_imgs = f_imgs["imgs_tested"]
    # 特征的group名根据cnn_name而定
    feature_name = str(cnn_name) + "_features"
    try:
        f_features = f_features.create_group(feature_name)
    except:
        f_features = f_features[feature_name]
    keys_dataset = list(f_features.keys())
    print("keys_dataset: ", keys_dataset)

    for i, key in enumerate(f_imgs.keys()):
        if key in keys_dataset:
            print("有重复的key")
            continue
        pics = f_imgs[key].value
        features = []
        for pic in pics:
            pic = np.expand_dims(pic, axis=0)
            feature = model.predict_on_batch(pic)
            feature = feature.reshape(-1)
            features.append(feature)
        features = np.asarray(features)
        try:
            f_features.create_dataset(str(key), data=features)
        except:
            del f_features[str(key)]
            f_features.create_dataset(str(key), data=features)

        print("-" * 20, i + 1)
        # print("features.shape:", features.shape)


def _main():
    # extract_cnn_features(cnn_name="VGG16", f_imgs=F_IMGS_1, f_features=F_FEATURES)
    # extract_cnn_features(cnn_name="VGG19", f_imgs=F_IMGS_1, f_features=F_FEATURES)
    extract_cnn_features(cnn_name="ResNet50", f_imgs=F_IMGS_1, f_features=F_FEATURES)
    extract_cnn_features(cnn_name="InceptionV3", f_imgs=F_IMGS_1, f_features=F_FEATURES)
    extract_cnn_features(cnn_name="Xception", f_imgs=F_IMGS_1, f_features=F_FEATURES)
    # extract_cnn_features(cnn_name="MobileNet", f_imgs=F_IMGS_1, f_features=F_FEATURES)


if __name__ == '__main__':
    F_IMGS_1 = h5py.File("/home/zj/helloworld/tr/offline/functions/imgs_preprocessed/imgs_preprocessed_300_300_3.h5",
                         "r")
    F_FEATURES = h5py.File("/home/zj/helloworld/tr/offline/functions/extract_cnn_features/cnn_features_300_trained.h5",
                           "a")
    begin = datetime.now()
    print("开始时间： ", begin)
    _main()
    end = datetime.now()
    time = (end - begin).seconds
    print("结束时间： ", end)
    print("总耗时: {}s".format(time))
