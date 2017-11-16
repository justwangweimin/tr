#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : h5py_test.py
# @Author: zjj421
# @Date  : 17-10-7
# @Desc  :
import h5py
from h5py import Dataset
from matplotlib import pyplot as plt

# bb860237fa71def388c4d8cb5c057768
def __main():
    f = h5py.File('/home/zj/helloworld/kaggle/threat_recognition/hdf5/cnn_features_200_tested.h5',
                  'r')
    # f = f["VGG19_features"]
    # print(f["0240c8f1e89e855dcd8f1fa6b1e2b944"])
    # print(f["0240c8f1e89e855dcd8f1fa6b1e2b944"].value)
    # del f["MobileNet"]
    print(len(f))
    # print("bb860237fa71def388c4d8cb5c057768" in f.keys())
    # for name in f:
    #     print(name)
    # print("*" * 20)
    for key in f.keys():
        print(key)
        print(f[key])
        # print(f[key].shape)
        print("-"*15)
    # f.visititems(func)
    # printname(f['predictions']['predictions'])
    # for name in f['predictions']['predictions'].keys():
    #     print(f['predictions']['predictions'][name].name)
    #     print(f['predictions']['predictions'][name].value)
    # printname("-+" * 25)
    # f.visit(printname)
    # print(f["imgs_trained"]["0240c8f1e89e855dcd8f1fa6b1e2b944"][0].shape)
    # print(type(f["imgs_trained"]["0240c8f1e89e855dcd8f1fa6b1e2b944"]))
    # plot_image_set(f["imgs_trained"]["0240c8f1e89e855dcd8f1fa6b1e2b944"])


def printname(name):
    print(name)


def func(name, obj):
    if isinstance(obj, Dataset):
        print(name)


# 显示.aps文件的所有16个视图
def plot_image_set(img):
    # show the graphs
    fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    i = 0
    for row in range(4):
        for col in range(4):
            print(type(img[i]))
            axarr[row, col].imshow(img[i], cmap='pink')
            i += 1
    plt.show()
    print('Done!')


if __name__ == '__main__':
    __main()
