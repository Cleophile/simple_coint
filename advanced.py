#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen

first_data = pd.read_csv("../future_by_date/20190612.csv")
instruments = list(set(first_data['windcode']))
#  n_instr = len(instruments)

future = first_data[first_data['windcode']==instruments[0]][['time','close']]
future.fillna(method='ffill',inplace=True)
future.rename(columns={'close':instruments[0]},inplace=True)

for i in instruments[1:]:
    data = first_data[first_data['windcode']==i][['time','close']]
    future = pd.merge(future,data,on='time')
    future.fillna(method='ffill',inplace=True)
    future.rename(columns={'close':i},inplace=True)

del future['time']
print(future)

intruments = future.columns
print(instruments)

#  jres = coint_johansen(future, det_order=0, k_ar_diff=1)
