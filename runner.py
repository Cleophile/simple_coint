#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

# afternoon only

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro

with open('./date_list.txt') as f:
    content = f.read()

date_list = json.loads(content)
money = 10000
money_step = [10000]

for date_index in range(1,len(date_list)):
    d_1 = date_list[date_index-1]
    d = date_list[date_index]
    first_data = pd.read_csv("../future_by_date/{}_close.csv".format(d_1))
    trade_set = set(first_data.columns)
    second_raw_data = pd.read_csv("../future_by_date/{}_close.csv".format(d))
    trade_set = trade_set & set(second_raw_data)
    second_data = second_raw_data[second_raw_data['time'] <= 1130]
    full_data = first_data.append(second_data)
    full_data = full_data[list(trade_set)].copy()
    full_data.fillna(method='ffill',inplace=True)

    if 'Unnamed: 0' in full_data.columns:
        del full_data['Unnamed: 0']
    del full_data['time']

    instruments = full_data.columns
    n_instr = len(instruments)

    clist = []
    for x in range(n_instr):
        for y in range(x+1,n_instr):
            i = instruments[x]
            j = instruments[y]
            _,result,*_ = coint(full_data[i],full_data[j])
            if result >= 0.05:
                continue
            clist.append([i,j,result])
    clist = sorted(clist,key=lambda x:x[2])
    #  print(clist)

    final_list = []
    for i,j,_ in clist:
        future0 = full_data[i].to_numpy().reshape(-1,1)
        future1 = full_data[j].to_numpy()

        reg = LinearRegression().fit(future0,future1)

        residue = future1 - reg.predict(future0)

        _, adftest_result, *_ = adfuller(residue)

        if adftest_result < 0.05:
            _, shapiro_result, *_ = shapiro(residue)
            #  print("p-value for normality test:", shapiro_result)
            if shapiro_result >= 0.05:
                #  print("Non normal residue found")
                pass
            else:
                # generate signal
                # TODO Add moving average?
                #  z_residue = (residue - np.mean(residue)) / np.std(residue)
                #  print(z_residue)
                final_list.append([i,j,reg,np.std(residue)])
        else:
            #  print("Non-stantionary cointegration detected!")
            #  print("p-value for ADF test:", adftest_result)
            pass

    trade_data = second_raw_data[second_raw_data['time']>=1341]
    #  print(final_list)

    for i,j,reg,std in final_list[:1]:
        future0 = trade_data[i].to_numpy().reshape(-1,1)
        n = len(future0)
        future1 = trade_data[j].to_numpy()
        potential_signal = (future1 - reg.predict(future0)) / std
        coef = reg.coef_[0]
        if coef >= -0.001 and coef <= 0.001:
            continue
        if coef < 0:
            continue
        # > 0 sell future1 buy future0 direction 1
        # < 0 buy future1 sell future0 direction -1
        p = 0
        direction = 0
        portfolio = []

        while(p < n - 30):
            price0 = future0[p]
            price1 = future1[p]
            if potential_signal[p] > 2.5:
                direction = 1
                if coef > 0:
                    combo = money / (price1 + coef*price0)
                    portfolio.append([price0,combo*coef,1])
                    portfolio.append([price1,combo,-1])
                else:
                    combo = money / (price1 - coef*price0)
                    portfolio.append([price0,-combo*coef,-1])
                    portfolio.append([price1,combo,-1])
                break

            if potential_signal[p] < -2.5:
                direction = -1
                if coef > 0:
                    combo = money / (price1 + coef*price0)
                    portfolio.append([price0,combo*coef,-1])
                    portfolio.append([price1,combo,1])
                else:
                    combo = money / (price1 - coef*price0)
                    portfolio.append([price0,-combo*coef,1])
                    portfolio.append([price1,combo,1])
                break
            p+=1
        p+=1
        if len(portfolio)==0:
            break
        #  print(portfolio)
        while(p < n - 1):
            new_money = money + portfolio[0][2] * (future0[p] - portfolio[0][0]) * portfolio[0][1] + portfolio[1][2] * (future1[p] - portfolio[1][0]) * portfolio[1][1]
            earn_ratio = (new_money - money) / money
            if earn_ratio > 0.02 or earn_ratio < -0.02:
                portfolio = []
                money = new_money
                break
            if direction==1 and potential_signal[p] < 0 and earn_ratio > 0:
                money = new_money
                portfolio = []
                break
            if direction==-1 and potential_signal[p] > 0 and earn_ratio > 0:
                money = new_money
                portfolio = []
                break
            p+=1

        if(portfolio!=[]):
            new_money = money + portfolio[0][2] * (future0[p] - portfolio[0][0]) * portfolio[0][1] + portfolio[1][2] * (future1[p] - portfolio[1][0]) * portfolio[1][1]
            portfolio = []
            money = new_money
        money_step.append(money)

plt.plot(money_step)
plt.show()
