# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15, 16:50 2021

Utility functions that computes the "Discriminant ratio" or "D ratio"
This file presents an abstract of the code used for prepare a publication
Use without prior reading the article might be misleading or inadequate    

@author: JDE65 (Github)
j.dessain@navagne.com   ///  j.dessain@ieseg.fr
www.navagne.com
All rights reserved - Copyright Navagne (2021)
"""

###=== 0. Installing libraries ============
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, skew, kurtosis, shapiro

###=== 1. Functions
## d_ratios = get_d(return_bh, return_pred))
## Inputs for all functions include:
#   1. log_return = numpy array of size (n, ) with the daily log return of a financial asset
#   2. asset_value = reference for VaR computation. Here 100
#   3. confid = confidence level for Cornish-Fisher Value-at-Risk. Usually 1% or 5% - here at 1%
#   4.a return_bh  = numpy array of size (n, ) with the daily log return of the Buy & HOLD strategy
#   4.b return_pred = numpy array of size (n, ) with the daily log return of a financial asset

## Outputs
# Output d_ratios is the numpy array d_ratios of size (7,3) with the output of the D ratio computation
## 0 = d_ratio, d_ratio1 for first period & d_ratio2 for 2nd period
## 1 = roi_bh : RoI for BH for full period, period 1 and period 2
## 2 = roi_pred RoI for pred for full period, period 1 and period 2
## 3 = varbh : CF-VaR of BH for full period, period 1 and period 2
## 4 = varpred : CF-VaR of pred for full period, period 1 and period 2
## 5 = rtv_bh : 'Return-to-VaR' of BH for full period, period 1 and period 2
## 6 = rtv_pred : 'Return-to-VaR' of pred for full period, period 1 and period 2

### 1.1 Skewness & Kurtosis
def risk_skew_kurt(log_return):
    mean = np.mean(log_return)
    st_dev = np.std(log_return)
    skew = stats.skew(log_return)
    kurt = stats.kurtosis(log_return)
    return (mean, st_dev), (skew, kurt)

### 1.2 Cornish Fisher VaR
def risk_cf_exp_var(log_return, asset_value, confid):
    (mean, st_dev), (skew, kurt) = risk_skew_kurt(log_return)
    quantile = norm.ppf(confid)
    cf_exp = quantile + (quantile ** 2 - 1) * skew / 6 
    + (quantile ** 3 - 3 * quantile) * (kurt - 3) / 24 
    - (2 * quantile ** 3 - 5 * quantile) * (skew ** 2) / 36
    cf_var = mean + st_dev * cf_exp
    cf_asset_value = asset_value * (1 + cf_var)
    return cf_exp, cf_var, cf_asset_value

### 1.3 D ratio
def get_d(return_bh, return_pred):
    d_ratios = np.zeros((7, 3))
    
    period1 = int(len(return_bh)/2)  ## mid-period for computing D ratio stability
    cf_exp, varbh, _ = risk_cf_exp_var(return_bh[1:, ], 100, 0.01)
    cf_exp, varbh_1, _ = risk_cf_exp_var(return_bh[1:period1, ], 100, 0.01)   ## CF-VaR BH period 1
    cf_exp, varbh_2, _ = risk_cf_exp_var(return_bh[period1:, ], 100, 0.01)   ## CF-VaR BH period 2
    cf_exp, varpred, _ = risk_cf_exp_var(return_pred[1:, ], 100, 0.01)  
    cf_exp, varpred_1, _ = risk_cf_exp_var(return_pred[1:period1, ], 100, 0.01)    ## CF-VaR Pred period 1
    cf_exp, varpred_2, _ = risk_cf_exp_var(return_pred[period1:, ], 100, 0.01)    ## CF-VaR Pred period 2
    d_ratios[1, 0] = np.mean(return_bh[1:, ], axis = 0) * 252 
    d_ratios[1, 1] = np.mean(return_bh[1:period1, ], axis = 0) * 252 
    d_ratios[1, 2] = np.mean(return_bh[period1:, ], axis = 0) * 252
    d_ratios[2, 0] = np.mean(return_pred[1:, ], axis = 0) * 252
    d_ratios[2, 1] = np.mean(return_pred[1:period1, ], axis = 0) * 252
    d_ratios[2, 2] = np.mean(return_pred[period1:, ], axis = 0) * 252 
    d_ratios[3, 0] = varbh
    d_ratios[3, 1] = varbh_1
    d_ratios[3, 2] = varbh_2
    d_ratios[4, 0] = varpred
    d_ratios[4, 1] = varpred_1
    d_ratios[4, 2] = varpred_2
    d_ratios[5, 0] = d_ratios[1, 0] / -d_ratios[3, 0]       ## rtv_bh
    d_ratios[5, 1] = d_ratios[1, 1] / -d_ratios[3, 1]       ## rtv_bh1
    d_ratios[5, 2] = d_ratios[1, 2] / -varbh_2              ## rtv_bh2
    d_ratios[6, 0] = d_ratios[2, 0] / -varpred              ## rtv_pred
    d_ratios[6, 1] = d_ratios[2, 1] / -varpred_1            ## rtv_pred1
    d_ratios[6, 2] = d_ratios[2, 2] / -varpred_2            ## rtv_pred2
    ## Final step : compute D ratio = 1 + (rtv_pred - rtv_bh) / ABS(rtv_bh)
    d_ratios[0, 0] = 1 + (d_ratios[6, 0]  - d_ratios[5, 0]) / np.abs(d_ratios[5, 0])
    d_ratios[0, 1] = 1 + (d_ratios[6, 1]  - d_ratios[5, 1]) / np.abs(d_ratios[5, 1])
    d_ratios[0, 2] = 1 + (d_ratios[6, 2]  - d_ratios[5, 2]) / np.abs(d_ratios[5, 2])
    return d_ratios

