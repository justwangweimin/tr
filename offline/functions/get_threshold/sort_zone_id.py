#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : sort_zone_id.py
# @Author: zjj421
# @Date  : 17-10-15
# @Desc  :

import os
import numpy as np
import pandas as pd


def sort_zone_id(csvfile, save_ouput=False):
    df = pd.read_csv(csvfile)
    df["Subject_id"], df["Zone_id"] = df["Id"].str.split("_Zone", 1).str
    df["Zone_id"] = df["Zone_id"].apply(lambda x: int(x))
    df = df.sort_values(by=["Subject_id", "Zone_id"])
    # del df["Subject_id"], df["Zone_id"]
    # df = df[["Id", "Probability"]]
    df.index = np.arange(len(df["Id"]))
    if save_ouput:
        dirname, basename = os.path.split(csvfile)
        new_basename = "sorted_" + basename
        new_csvfile = os.path.join(dirname, new_basename)
        df.to_csv(new_csvfile,
                  index=False, sep=",")
    return df

if __name__ == '__main__':
    CSVFILE = "./stage1_labels.csv"
    sort_zone_id(CSVFILE)
