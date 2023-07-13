#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats
from tqdm import tqdm
import math
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm


# In[3]:


def turningpoints(lst):
    dx = np.diff(lst)
    return np.sum(dx[1:] * dx[:-1] < 0)


# In[4]:


def autocorrelation_f1(time_series):
    autocorrelation_ts = acf(time_series, nlags=5)
    return autocorrelation_ts


# In[5]:


def partial_autocorrelation_f2(time_series):
    partial_autocorrelation_ts = pacf(time_series, nlags=5)
    return partial_autocorrelation_ts


# In[6]:


def variance_f3(time_series):
    variance_ts = time_series.std()
    return variance_ts


# In[7]:


def skewness_f4(time_series):
    variance_ts = time_series.skew()
    return variance_ts


# In[8]:


def kurtosis_f5(time_series):
    kurtosis_ts = time_series.kurtosis()
    return kurtosis_ts


# In[9]:


def turning_points_f6(time_series):
    turning_points_ts = turningpoints(time_series)
    return turning_points_ts


# In[10]:


# The bicorrelations at the first three lags were used
def three_point_autocorrelation_f7(time_series, lag_delay=3):
    three_point_autocorrelation = acorr = sm.tsa.acf(time_series, nlags = lag_delay)
    return three_point_autocorrelation[:-1]


# In[11]:


# The mutual information at the first three lags were used
def mutual_info_f8(time_series, lag_delay=3):
    mutual_info_ts = []
    for _ in range(0, lag_delay+1):
        lagged_time_series = time_series[:-lag_delay]
        time_series_ = time_series[lag_delay:].to_numpy().reshape(-1, 1)
        mutual_info = mutual_info_regression(time_series_, lagged_time_series)
        mutual_info_ts.append(mutual_info[0])
    return mutual_info_ts


# In[12]:


def feature_extraction(time_series):
    time_series_diff = time_series - time_series.shift()
    time_series_diff = time_series_diff[1:]
    
    features = []

    # autocorrelation F1
    autocorrelation_ts = autocorrelation_f1(time_series_diff)
    # partial autocorrelation F2
    partial_autocorrelation_ts = partial_autocorrelation_f2(time_series_diff)
    # variance F3
    variance_ts = variance_f3(time_series_diff)
    # skewness F4
    skewness_ts = skewness_f4(time_series_diff)
    # Kurtoisis F5
    kurtoisis_ts = kurtosis_f5(time_series_diff)
    # Turning Point F6
    turning_point_ts = turning_points_f6(time_series_diff)
    # Three point autocorrelation F7
    three_point_autocorrelation_ts = three_point_autocorrelation_f7(time_series_diff)
    # Mutual info F8
    mutual_info_ts = mutual_info_f8(time_series_diff)

    for i in autocorrelation_ts:
        features.append(i)

    for i in partial_autocorrelation_ts:
        features.append(i)

    features.append(variance_ts)

    features.append(skewness_ts)

    features.append(kurtoisis_ts)

    features.append(turning_point_ts)
    
    for i in three_point_autocorrelation_ts:
        features.append(i)

    for i in mutual_info_ts:
        features.append(i)
    
    return features


# In[ ]:




