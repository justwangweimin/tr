#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_image.py
# @Author: zjj421
# @Date  : 17-8-27
# @Desc  :

from __future__ import print_function
from __future__ import division
import numpy as np
import os

from PIL import Image
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import scipy.stats as stats

# For every scan in the dataset, you will be predicting the
# probability that a threat is present in each of 17 body zones.




# ----------------------------------------------------------------------------------
# read_header(infile):  takes an aps file and creates a dict of the data
#
# infile:               an aps file
#
# returns:              all of the fields in the header
# ----------------------------------------------------------------------------------

def read_header(infile):
    # declare dictionary
    h = dict()

    with open(infile, 'r+b') as fid:
        h['filename'] = b''.join(np.fromfile(fid, dtype='S1', count=20))
        h['parent_filename'] = b''.join(np.fromfile(fid, dtype='S1', count=20))
        h['comments1'] = b''.join(np.fromfile(fid, dtype='S1', count=80))
        h['comments2'] = b''.join(np.fromfile(fid, dtype='S1', count=80))
        h['energy_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['config_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['file_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['trans_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scan_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['data_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['date_modified'] = b''.join(np.fromfile(fid, dtype='S1', count=16))
        h['frequency'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['mat_velocity'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['num_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_polarization_channels'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['spare00'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['adc_min_voltage'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['adc_max_voltage'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['band_width'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['spare01'] = np.fromfile(fid, dtype=np.int16, count=5)
        h['polarization_type'] = np.fromfile(fid, dtype=np.int16, count=4)
        h['record_header_size'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['word_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['word_precision'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['min_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['max_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['avg_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['data_scale_factor'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['data_units'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['surf_removal'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['edge_weighting'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['x_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['y_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['z_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['t_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['spare02'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['x_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['scan_orientation'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scan_direction'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['data_storage_order'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scanner_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['x_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['t_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['num_x_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_y_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_z_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_t_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['x_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['date_processed'] = b''.join(np.fromfile(fid, dtype='S1', count=8))
        h['time_processed'] = b''.join(np.fromfile(fid, dtype='S1', count=8))
        h['depth_recon'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['elevation_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['roll_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['azimuth_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['adc_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['spare06'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scanner_radius'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['t_delay'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['range_gate_start'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['range_gate_end'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['ahis_software_version'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['spare_end'] = np.fromfile(fid, dtype=np.float32, count=10)

    return h


# # unit test ----------------------------------
# header = read_header(APS_FILE_NAME)
#
# for data_item in sorted(header):
#     print('{} -> {}'.format(data_item, header[data_item]))


# ----------------------------------------------------------------------------------
# read_data(infile):  reads and rescales any of the four image types
#
# infile:             an .aps, .aps3d, .a3d, or ahi file
#
# returns:            the stack of images
#
# note:               word_type == 7 is an np.float32, word_type == 4 is np.uint16
# ----------------------------------------------------------------------------------

def read_data(infile):
    # read in header and get dimensions
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])

    extension = os.path.splitext(infile)[1]

    with open(infile, 'rb') as fid:

        # skip the header
        fid.seek(512)

        # handle .aps and .a3aps files
        if extension == '.aps' or extension == '.a3daps':

            if (h['word_type'] == 7):
                data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)

            elif (h['word_type'] == 4):
                data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, ny, nt, order='F').copy()

        # handle .a3d files
        elif extension == '.a3d':

            if (h['word_type'] == 7):
                data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)

            elif (h['word_type'] == 4):
                data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, nt, ny, order='F').copy()

            # handle .ahi files
        elif extension == '.ahi':
            data = np.fromfile(fid, dtype=np.float32, count=2 * nx * ny * nt)
            data = data.reshape(2, ny, nx, nt, order='F').copy()
            real = data[0, :, :, :].copy()
            imag = data[1, :, :, :].copy()

        # print(nx, ny, nt)
        if extension != '.ahi':
            return data
        else:
            return real, imag


# unit test ----------------------------------
# d = read_data(APS_FILE_NAME)
# print(d.shape)
# print(type(d))
# img = d.transpose()
# print(img.shape)


# ----------------------------------------------------------------------------------
# plot_image_set(infile):  takes an aps file and shows all 16 90 degree shots
#
# infile:                  an aps file
# ----------------------------------------------------------------------------------
def plot_image_set1(infile):
    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)

    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()

    # show the graphs
    fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    i = 0
    for row in range(4):
        for col in range(4):
            print(img[i].shape)
            if i ==0:
                img[i] = convert_to_grayscale(img[i])
                cv2.imwrite("/home/zj/桌面/test01.png", np.flipud(img[i]))
            resized_img = cv2.resize(img[i],(150,150), fx=0.1, fy=0.1)
            print(resized_img.shape)
            axarr[row, col].imshow(np.flipud(resized_img), cmap=COLORMAP)
            if(i == 0):
                resized_img = convert_to_grayscale(resized_img)
                cv2.imwrite("/home/zj/桌面/test.png", np.flipud(resized_img))
            i += 1

    print('Done!')




# 显示.aps文件的所有16个视图
def plot_image_set2(infile):
    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)

    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()

    # show the graphs
    fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    i = 0
    for row in range(4):
        for col in range(4):
            img_t = img[i]
            # img_t = convert_to_grayscale(img_t)
            # img_t = Image.fromarray(img_t)
            # print(img_t.getbands())
            # print("单张图片处理前的shape: {}".format(img[i].shape))
            # resized_img = cv2.resize(img[i], (0, 0), fx=0.1, fy=0.1)
            # resized_img = cv2.resize(img[i], (150, 150), fx=0.1, fy=0.1)
            # resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
            axarr[row, col].imshow(np.flipud(img_t), cmap=COLORMAP)
            # axarr[row, col].imshow(resized_img.transpose(), cmap=COLORMAP)
            # print("单张图片处理后的shape: {}".format(resized_img.shape))
            # resized_img = Image.fromarray(resized_img)
            # print(resized_img.getbands())
            i += 1

    print('Done!')


# unit test ----------------------------------




# ----------------------------------------------------------------------------------
# get_single_image(infile, nth_image):  returns the nth image from the image stack
#
# infile:                              an aps file
#
# returns:                             an image
# ----------------------------------------------------------------------------------

# 获取单一.aps文件的第n张视图
def get_single_image(infile, nth_image):
    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)

    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()

    # print(img.shape)

    # return np.flipud(img[nth_image])
    return img[nth_image]


# unit test ---------------------------------------------------------------
# an_img1 = get_single_image(APS_FILE_NAME, 0)
# print(type(an_img1))
# print(an_img1.shape)
#
# an_img2 = np.flipud(an_img1)
# print(type(an_img2))
# print(an_img2.shape)
#
# an_img3 = convert_to_grayscale(an_img2)
# print(type(an_img3))
# print(an_img3.shape)
#
#
# # an_img4 = an_img2.copy()
# an_img4 = cv2.cvtColor(an_img3, cv2.COLOR_GRAY2RGB)
# print(type(an_img4))
# print(an_img4.shape)
#
# an_img5 = cv2.resize(an_img4, (150,150), interpolation=cv2.INTER_AREA)  # return a new picture
# print(type(an_img5))
# print(an_img5.shape)
# # print(an_img.filename)
# # 判断图片是否为RGB图片
# # im = Image.open(filename_path)
# # if not im.getbands() == ('R', 'G', 'B'):
# #     num_failure += 1
# #     print("{} 不是RGB图片".format(filename_path))
# #     continue
#
# fig, axarr = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
#
# axarr[0].imshow(an_img2, cmap=COLORMAP)
# axarr[1].imshow(an_img3, cmap=COLORMAP)
# axarr[2].imshow(an_img4, cmap=COLORMAP)
# axarr[3].imshow(an_img5, cmap=COLORMAP)
# # plt.subplot(122)
# # plt.hist(an_img.flatten(), bins=256, color='c')
# # plt.xlabel("Raw Scan Pixel Value")
# # plt.ylabel("Frequency")
# plt.show()


# ----------------------------------------------------------------------------------
# convert_to_grayscale(img):           converts a ATI scan to grayscale
#
# infile:                              an aps file
#
# returns:                             an image
# ----------------------------------------------------------------------------------

# 把图片转化为灰度图片　Rescaling the Image
def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled)


# # unit test ------------------------------------------
# img_rescaled = convert_to_grayscale(an_img)
#
# fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
#
# axarr[0].imshow(img_rescaled, cmap=COLORMAP)
# plt.subplot(122)
# plt.hist(img_rescaled.flatten(), bins=256, color='c')
# plt.xlabel("Grayscale Pixel Value")
# plt.ylabel("Frequency")
# plt.show()


# -------------------------------------------------------------------------------
# spread_spectrum(img):        applies a histogram equalization transformation
#
# img:                         a single scan
#
# returns:                     a transformed scan
# -------------------------------------------------------------------------------

# 传播光谱
# spread the spectrum to improve contrast
def spread_spectrum(img):
    img = stats.threshold(img, threshmin=12, newval=0)

    # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img


# # unit test ------------------------------------------
# img_high_contrast = spread_spectrum(img_rescaled)
#
# fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
#
# axarr[0].imshow(img_high_contrast, cmap=COLORMAP)
# plt.subplot(122)
# plt.hist(img_high_contrast.flatten(), bins=256, color='c')
# plt.xlabel("Grayscale Pixel Value")
# plt.ylabel("Frequency")
# plt.show()

if __name__ == '__main__':
    # constants
    COLORMAP = 'pink'
    # APS_FILE_NAME = '/media/zj/study/kaggle/sample/00360f79fd6e02781457eda48f85da90.aps'
    # APS_FILE_NAME = '/media/zj/study/kaggle/sample/0043db5e8c819bffc15261b1f1ac5e42.a3d'
    PATH = "/home/zj/helloworld/kaggle/tr/input/sample/0043db5e8c819bffc15261b1f1ac5e42.a3daps"

    # BODY_ZONES = '/media/zj/study/kaggle/body_zones.png'
    # THREAT_LABELS = '/media/zj/study/kaggle/stage1_labels.csv'
    plot_image_set1(PATH)
    plt.show()