#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : image_resize.py
# @Author: zjj421
# @Date  : 17-8-28
# @Desc  :


import cv2


def image_resize(img):
    if type(img) == str:
        img = cv2.imread(img)
    print("原图片尺寸：", img.shape)
    # height, width = img.shape[:2]
    # 缩小图像
    size = (150, 150)
    shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # return a new picture
    print("改变后的图片尺寸：", shrink.shape)
    return shrink
    # 放大图像
    # fx = 1.6
    # fy = 1.2
    # enlarge = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    # 显示
    # cv2.imshow("src", img)
    # cv2.imshow("shrink", shrink)
    # cv2.imshow("enlarge", enlarge)
    # cv2.waitKey(0)


if __name__ == '__main__':
    IMG = "/home/zj/桌面/DNC_image.jpg"
    image_resize(IMG)
