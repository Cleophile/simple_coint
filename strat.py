#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

first_data = pd.read_csv("../future_by_date/20190612.csv")
instruments = list(set(first_data['windcode']))
n_instr = len(instruments)

def extract_date(s):
    t = s['Unnamed: 0']
    return t.year*10000 + t.month*100 + t.day

def extract_time(s):
    t = s['Unnamed: 0']
    return t.hour*100 + t.minute

# check for conintegration pairs
clist = []
for x in range(n_instr):
    for y in range(x+1,n_instr):
        i = instruments[x]
        j = instruments[y]
        #  future = pd.DataFrame()
        #  print("Doing:",i,j)
        future0 = first_data[first_data['windcode']==i][['time','close']]
        #  future0.columns=['time0','close0']
        future0.fillna(method='ffill',inplace=True)
        future1 = first_data[first_data['windcode']==j][['time','close']]
        future1.fillna(method='ffill',inplace=True)
        #  future1.columns=['time1','close1']
        # max_lag: 0
        # if set to 1 or larger, more of them can pass the test
        result = coint(future0['close'],future1['close'])
        clist.append((i,j,result[1]))

for i in clist:
    if i[2] < 0.05:
        print(i)

#  test and plot area
#  i,j = 'SR909.CZC', 'AL1905.SHF'
#  i,j = 'PP1905.DCE', 'A1905.DCE'
#  future0 = first_data[first_data['windcode']==i][['time','close']]
#  future1 = first_data[first_data['windcode']==j][['time','close']]
#  future0.fillna(method='ffill',inplace=True)
#  future1.fillna(method='ffill',inplace=True)

#  d0 = future0['close'].to_numpy()
#  d1 = future1['close'].to_numpy()

#  d0 -= np.average(d0)
#  d1 -= np.average(d1)

#  d0 /= np.std(d0)
#  d1 /= np.std(d1)

#  plt.plot(d0)
#  plt.plot(d1)

# add one section selecting pairs

i, j = 'AL1908.SHF', 'P1909.DCE'

future0 = first_data[first_data['windcode']==i]['close'].copy()
future1 = first_data[first_data['windcode']==j]['close'].copy()
future0.fillna(method='ffill',inplace=True)
future1.fillna(method='ffill',inplace=True)

future0 = future0.to_numpy().reshape(-1,1)
future1 = future1.to_numpy()

reg = LinearRegression().fit(future0,future1)
print("R^2:", reg.score(future0,future1))
print("coef:", reg.coef_, "intercept:", reg.intercept_)

residue = future1 - reg.predict(future0)

#  print("------------------------ Residue ------------------------")
#  print(residue)
#  print("------------------------ Residue ------------------------")

_,adftest_result,*_ = adfuller(residue)
print("p-value for ADF test:", adftest_result)

if adftest_result < 0.05:
    pass
else:
    print("Non-stantionary cointegration detected!")

#  test plot of residue
#  plt.plot(residue)
#  plt.show()

# 收益补充
