#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : reading_images.py
# @Author: zjj421
# @Date  : 17-11-5
# @Desc  :


import numpy as np
import os
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.animation


def read_header(infile):
    """Read image header (first 512 bytes)
    """
    h = dict()
    fid = open(infile, 'r+b')
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


def read_data(infile):
    """Read any of the 4 types of image files, returns a numpy array of the image contents
    """
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512)  # skip header
    if extension == '.aps' or extension == '.a3daps':
        if (h['word_type'] == 7):  # float32
            data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)
        elif (h['word_type'] == 4):  # uint16
            data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)
        data = data * h['data_scale_factor']  # scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy()  # make N-d image
    elif extension == '.a3d':
        if (h['word_type'] == 7):  # float32
            data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)
        elif (h['word_type'] == 4):  # uint16
            data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)
        data = data * h['data_scale_factor']  # scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy()  # make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype=np.float32, count=2 * nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0, :, :, :].copy()
        imag = data[1, :, :, :].copy()
    fid.close()
    if extension != '.ahi':
        return data
    else:
        return real, imag


# matplotlib.rc('animation', html='html5')


def plot_image(path):
    data = read_data(path)
    fig = matplotlib.pyplot.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)

    def animate(i):
        im = ax.imshow(np.flipud(data[:, :, i].transpose()), cmap='viridis')
        return [im]

    return matplotlib.animation.FuncAnimation(fig, animate, frames=range(0, data.shape[2]), interval=200, blit=True)


import cv2


def dump_to_video(dbpath, video_path):
    data = read_data(dbpath)
    w, h, n = data.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (w, h))

    for i in range(n):
        img = np.flipud(data[:, :, i].transpose())
        norm = plt.Normalize()
        img = norm(img)
        img = plt.cm.viridis(img)
        img = (255.0 * img).astype(np.uint8)
        out.write(img)

    out.release()


def show_imgs(data_root):
    for root, dirs, files in os.walk(data_root):
        for i, file in enumerate(files):
            file_path = os.path.join(root, file)
            anm = plot_image(file_path)
            matplotlib.pyplot.show()
            print("已显示 ---{}--- 张图片.".format(i + 1))


if __name__ == '__main__':
    # anm = plot_image("/home/zj/helloworld/kaggle/tr/input/sample/00360f79fd6e02781457eda48f85da90.aps")
    DATA_ROOT = "/media/zj/study/kaggle/syblink_stage1_aps/syblink_test_data_root"
    show_imgs(DATA_ROOT)
    # dump_to_video("/home/zj/helloworld/kaggle/tr/input/sample/00360f79fd6e02781457eda48f85da90.a3d", "00360f79fd6e02781457eda48f85da90_a3d.avi")
