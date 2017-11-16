#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : convert_gray_to_rgb.py
# @Author: zjj421
# @Date  : 17-8-29
# @Desc  :

import os
import cv2
file_input_path = ''
img = cv2.imread(file_input_path, 0)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
out_file = ''
cv2.imwrite(out_file, img)

