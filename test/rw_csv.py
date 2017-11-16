#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : rw_csv.py
# @Author: zjj421
# @Date  : 17-9-16
# @Desc  :

import pandas as pd
import numpy as np

# a = [1, 2, 3]
# a = np.asarray(a)
# b = ["z1", "z2", "z3"]
# dataframe = pd.DataFrame({"id": b, "probability": a})
# dataframe.to_csv("/tmp/testcsv.csv", index=False, sep=",")

# df = pd.read_csv("/tmp/testcsv.csv")
# print(type(df))
# test = "test"
# df[test] = [111, 222, 333]
# print(df)
# df.to_csv("/tmp/testcsv.csv", index=False)

df = pd.DataFrame()
print(type(df))
df["test"] = [1, 2, 3]
print(df)
