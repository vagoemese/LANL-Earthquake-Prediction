# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:41:42 2019

@author: Emi
"""
from featuretools.primitives import make_agg_primitive
import numpy as np
import scipy.stats as stats
import pandas as pd
from featuretools.variable_types import Numeric

def my_primitives():
    
    def gmean(x):
        return stats.gmean(np.absolute(list(filter(lambda a: a != 0, x))))

    def hmean(x):
        return stats.hmean(np.absolute(list(filter(lambda a: a != 0, x))))
    
    def kstatvar1(x):
        return stats.kstatvar(x, 1)
    
    def kstat2(x):
        return stats.kstat(x, 2)
    
    def kstatvar2(x):
        return stats.kstatvar(x, 2)
    
    def kstat3(x):
        return stats.kstat(x, 3)
    
    def kstat4(x):
        return stats.kstat(x, 4)
    
    def avg_change(x):
        return np.mean(np.diff(x))
    
    def avg_change_rate(x):
        return np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    
    def range_func(x):
        return max(x)-min(x)
    
    def std_first_50000(x):
        return x[:50000].std()
    
    def std_last_50000(x):
        return x[-50000:].std()
    
    def std_first_10000(x):
        return x[:10000].std()
    
    def std_last_10000(x):
        return x[-10000:].std()
    
    def avg_first_50000(x):
        return x[:50000].mean()
    
    def avg_last_50000(x):
        return x[-50000:].mean()
    
    def avg_first_10000(x):
        return x[:10000].mean()
    
    def avg_last_10000(x):
        return x[-10000:].mean()
    
    def min_first_50000(x):
        return x[:50000].min()
    
    def min_last_50000(x):
        return x[-50000:].min()
    
    def min_first_10000(x):
        return x[:10000].min()
    
    def min_last_10000(x):
        return x[-10000:].min()
    
    def max_first_50000(x):
        return x[:50000].max()
    
    def max_last_50000(x):
        return x[-50000:].max()
    
    def max_first_10000(x):
        return x[:10000].max()
    
    def max_last_10000(x):
        return x[-10000:].max()
    
    def max_to_min(x):
        return x.max() / np.abs(x.min())
    
    def count_big(x):
        return len(x[np.abs(x) > 500])
    
    def sum_func(x):
        return x.sum()
    
    def avg_change_rate_first_50000(x):
        return np.mean(np.nonzero((np.diff(x[:50000]) / x[:50000][:-1]))[0])
    
    def avg_change_rate_last_50000(x):
        return np.mean(np.nonzero((np.diff(x[-50000:]) / x[-50000:][:-1]))[0])
    
    def avg_change_rate_first_10000(x):
        return np.mean(np.nonzero((np.diff(x[:10000]) / x[:10000][:-1]))[0])
    
    def avg_change_rate_last_10000(x):
        return np.mean(np.nonzero((np.diff(x[-10000:]) / x[-10000:][:-1]))[0])
    
    def q95(x):
        return np.quantile(x, 0.95)
    
    def q99(x):
        return np.quantile(x, 0.99)
    
    def q05(x):
        return np.quantile(x, 0.05)
    
    def q01(x):
        return np.quantile(x, 0.01)
    
    def abs_q95(x):
        return np.quantile(np.abs(x), 0.95)
    
    def abs_q99(x):
        return np.quantile(np.abs(x), 0.99)
    
    def add_trend_feature(arr, abs_values=False):
        idx = np.array(range(len(arr)))
        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)
        return lr.coef_[0]
    
    def add_trend_feature_abs(arr):
        idx = np.array(range(len(arr)))
        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), np.abs(arr))
        return lr.coef_[0]

    def abs_mean(x):
        return np.abs(x).mean()
    
    def abs_std(x):
        return np.abs(x).std()
    
    def mad(x):
        return x.mad()
    
    def kurt(x):
        return x.kurtosis()
    
    def skew(x):
        return x.skew()
    
    def med(x):
        return x.median()
    
    def Hilbert_mean(x):
        return np.abs(hilbert(x)).mean()
    
    def Hann_window_mean(x):
        return (np.convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
    
    def classic_sta_lta(x, length_sta, length_lta):
        sta = np.cumsum(x ** 2)
        # Convert to float
        sta = np.require(sta, dtype=np.float)
        # Copy for LTA
        lta = sta.copy()
        # Compute the STA and the LTA
        sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
        sta /= length_sta
        lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
        lta /= length_lta
        # Pad zeros
        sta[:length_lta - 1] = 0
        # Avoid division by zero by setting zero values to tiny float
        dtiny = np.finfo(0.0).tiny
        idx = lta < dtiny
        lta[idx] = dtiny
        return sta / lta
    
    def classic_sta_lta1_mean(x):
        return  classic_sta_lta(x, 500, 10000).mean()

    def classic_sta_lta2_mean(x):
        return classic_sta_lta(x, 5000, 100000).mean()
    
    def classic_sta_lta3_mean(x):
        return classic_sta_lta(x, 3333, 6666).mean()
    
    def classic_sta_lta4_mean(x):
        return classic_sta_lta(x, 10000, 25000).mean()
    
    def Moving_average_700_mean(x):
        return x.rolling(window=700).mean().mean(skipna=True)
    
    def Moving_average_1500_mean(x):
        return x.rolling(window=1500).mean().mean(skipna=True)
    
    def Moving_average_3000_mean(x):
        return x.rolling(window=3000).mean().mean(skipna=True)
    
    def Moving_average_6000_mean(x):
        return x.rolling(window=6000).mean().mean(skipna=True)
    
    def exp_Moving_average_300_mean(x):
        return (pd.Series.ewm(x, span=300).mean()).mean(skipna=True)
    
    def exp_Moving_average_3000_mean(x):
        return (pd.Series.ewm(x, span=3000).mean()).mean(skipna=True)
    
    def exp_Moving_average_30000_mean(x):
        return (pd.Series.ewm(x, span=30000).mean()).mean(skipna=True)

    def iqr(x):
        return np.subtract(*np.percentile(x, [75, 25]))
    
    def q999(x):
        return np.quantile(x, 0.999)
    
    def q001(x):
        return np.quantile(x, 0.001)
    
    def ave10(x):
        return  stats.trim_mean(x, 0.1)  
    
    def ave_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return x_roll_std.mean()
    
    def std_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return x_roll_std.std()
    
    def max_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return x_roll_std.max()
    
    def min_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return x_roll_std.min()

    def q01_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return np.quantile(x_roll_std, 0.01)
    
    def q05_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return np.quantile(x_roll_std, 0.05)
    
    def q95_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return np.quantile(x_roll_std, 0.95)
    
    def q99_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return np.quantile(x_roll_std, 0.99)

    def av_change_abs_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return np.mean(np.diff(x_roll_std))
    
    def av_change_rate_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
    
    def abs_max_roll_std_10(x):
        x_roll_std = x.rolling(10).std().dropna().values
        return  np.abs(x_roll_std).max()
    
    def std_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return x_roll_mean.std()

    def max_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return x_roll_mean.max()
    
    def min_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return x_roll_mean.min()
    
    def q01_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return np.quantile(x_roll_mean, 0.01)
    
    def q05_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return np.quantile(x_roll_mean, 0.05)

    def q95_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return np.quantile(x_roll_mean, 0.95)
    
    def q99_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return np.quantile(x_roll_mean, 0.99)
    
    def av_change_abs_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return np.mean(np.diff(x_roll_mean))
    
    def av_change_rate_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

    def abs_max_roll_mean_10(x):
        x_roll_mean = x.rolling(10).mean().dropna().values
        return np.abs(x_roll_mean).max()
    
    def ave_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return x_roll_std.mean()
    
    def std_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return x_roll_std.std()
    
    def max_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return x_roll_std.max()
    
    def min_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return x_roll_std.min()

    def q01_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return np.quantile(x_roll_std, 0.01)
    
    def q05_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return np.quantile(x_roll_std, 0.05)
    
    def q95_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return np.quantile(x_roll_std, 0.95)
    
    def q99_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return np.quantile(x_roll_std, 0.99)

    def av_change_abs_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return np.mean(np.diff(x_roll_std))
    
    def av_change_rate_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
    
    def abs_max_roll_std_100(x):
        x_roll_std = x.rolling(100).std().dropna().values
        return  np.abs(x_roll_std).max()
    
    def std_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return x_roll_mean.std()

    def max_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return x_roll_mean.max()
    
    def min_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return x_roll_mean.min()
    
    def q01_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return np.quantile(x_roll_mean, 0.01)
    
    def q05_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return np.quantile(x_roll_mean, 0.05)

    def q95_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return np.quantile(x_roll_mean, 0.95)
    
    def q99_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return np.quantile(x_roll_mean, 0.99)
    
    def av_change_abs_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return np.mean(np.diff(x_roll_mean))
    
    def av_change_rate_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

    def abs_max_roll_mean_100(x):
        x_roll_mean = x.rolling(100).mean().dropna().values
        return np.abs(x_roll_mean).max()
    
    def ave_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return x_roll_std.mean()
    
    def std_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return x_roll_std.std()
    
    def max_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return x_roll_std.max()
    
    def min_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return x_roll_std.min()

    def q01_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return np.quantile(x_roll_std, 0.01)
    
    def q05_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return np.quantile(x_roll_std, 0.05)
    
    def q95_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return np.quantile(x_roll_std, 0.95)
    
    def q99_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return np.quantile(x_roll_std, 0.99)

    def av_change_abs_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return np.mean(np.diff(x_roll_std))
    
    def av_change_rate_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
    
    def abs_max_roll_std_1000(x):
        x_roll_std = x.rolling(1000).std().dropna().values
        return  np.abs(x_roll_std).max()
    
    def std_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return x_roll_mean.std()

    def max_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return x_roll_mean.max()
    
    def min_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return x_roll_mean.min()
    
    def q01_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return np.quantile(x_roll_mean, 0.01)
    
    def q05_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return np.quantile(x_roll_mean, 0.05)

    def q95_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return np.quantile(x_roll_mean, 0.95)
    
    def q99_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return np.quantile(x_roll_mean, 0.99)
    
    def av_change_abs_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return np.mean(np.diff(x_roll_mean))
    
    def av_change_rate_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

    def abs_max_roll_mean_1000(x):
        x_roll_mean = x.rolling(1000).mean().dropna().values
        return np.abs(x_roll_mean).max()

    def ave_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return x_roll_std.mean()
    
    def std_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return x_roll_std.std()
    
    def max_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return x_roll_std.max()
    
    def min_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return x_roll_std.min()

    def q01_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return np.quantile(x_roll_std, 0.01)
    
    def q05_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return np.quantile(x_roll_std, 0.05)
    
    def q95_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return np.quantile(x_roll_std, 0.95)
    
    def q99_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return np.quantile(x_roll_std, 0.99)

    def av_change_abs_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return np.mean(np.diff(x_roll_std))
    
    def av_change_rate_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
    
    def abs_max_roll_std_10000(x):
        x_roll_std = x.rolling(10000).std().dropna().values
        return  np.abs(x_roll_std).max()
    
    def std_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return x_roll_mean.std()

    def max_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return x_roll_mean.max()
    
    def min_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return x_roll_mean.min()
    
    def q01_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return np.quantile(x_roll_mean, 0.01)
    
    def q05_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return np.quantile(x_roll_mean, 0.05)

    def q95_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return np.quantile(x_roll_mean, 0.95)
    
    def q99_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return np.quantile(x_roll_mean, 0.99)
    
    def av_change_abs_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return np.mean(np.diff(x_roll_mean))
    
    def av_change_rate_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

    def abs_max_roll_mean_10000(x):
        x_roll_mean = x.rolling(10000).mean().dropna().values
        return np.abs(x_roll_mean).max()
    
    kstat2_pr = make_agg_primitive(function = kstat2,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    kstatvar1_pr = make_agg_primitive(function = kstatvar1,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    kstatvar2_pr = make_agg_primitive(function = kstatvar2,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    kstat3_pr = make_agg_primitive(function = kstat3,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    kstat4_pr = make_agg_primitive(function = kstat4,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    gmean_pr = make_agg_primitive(function = gmean,
                              input_types = [Numeric],
                              return_type = Numeric)  
    
    hmean_pr = make_agg_primitive(function = hmean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_change_pr = make_agg_primitive(function = avg_change,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_change_rate_pr = make_agg_primitive(function = avg_change_rate,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    range_pr = make_agg_primitive(function = range_func,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_first_50000_pr = make_agg_primitive(function = std_first_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_last_50000_pr = make_agg_primitive(function = std_last_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_first_10000_pr = make_agg_primitive(function = std_first_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_last_10000_pr = make_agg_primitive(function = std_last_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_first_50000_pr = make_agg_primitive(function = avg_first_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_last_50000_pr = make_agg_primitive(function = avg_last_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_first_10000_pr = make_agg_primitive(function = avg_first_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_last_10000_pr = make_agg_primitive(function = avg_last_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_first_50000_pr = make_agg_primitive(function = min_first_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_last_50000_pr = make_agg_primitive(function = min_last_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_first_10000_pr = make_agg_primitive(function = min_first_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_last_10000_pr = make_agg_primitive(function = min_last_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_first_50000_pr = make_agg_primitive(function = max_first_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_last_50000_pr = make_agg_primitive(function = max_last_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_first_10000_pr = make_agg_primitive(function = max_first_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_last_10000_pr = make_agg_primitive(function = max_last_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_to_min_pr = make_agg_primitive(function = max_to_min,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    count_big_pr = make_agg_primitive(function = count_big,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    sum_func_pr = make_agg_primitive(function = sum_func,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_change_rate_first_50000_pr = make_agg_primitive(function = avg_change_rate_first_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_change_rate_last_50000_pr = make_agg_primitive(function = avg_change_rate_last_50000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_change_rate_first_10000_pr = make_agg_primitive(function = avg_change_rate_first_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    avg_change_rate_last_10000_pr = make_agg_primitive(function = avg_change_rate_last_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q95_pr = make_agg_primitive(function = q95,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q99_pr = make_agg_primitive(function = q99,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q05_pr = make_agg_primitive(function = q05,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q01_pr = make_agg_primitive(function = q01,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_q95_pr = make_agg_primitive(function = abs_q95,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_q99_pr = make_agg_primitive(function = abs_q99,
                              input_types = [Numeric],
                              return_type = Numeric)

    trend_pr = make_agg_primitive(function = add_trend_feature,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_trend_pr = make_agg_primitive(function = add_trend_feature_abs,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_mean_pr = make_agg_primitive(function = abs_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_std_pr = make_agg_primitive(function = abs_std,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    mad_pr = make_agg_primitive(function = mad,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    kurt_pr = make_agg_primitive(function = kurt,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    skew_pr = make_agg_primitive(function = skew,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    med_pr = make_agg_primitive(function = med,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    Hilbert_mean_pr = make_agg_primitive(function = Hilbert_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    Hann_window_mean_pr = make_agg_primitive(function = Hann_window_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    classic_sta_lta1_mean_pr = make_agg_primitive(function = classic_sta_lta1_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    classic_sta_lta2_mean_pr = make_agg_primitive(function = classic_sta_lta2_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    classic_sta_lta3_mean_pr = make_agg_primitive(function = classic_sta_lta3_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    classic_sta_lta4_mean_pr = make_agg_primitive(function = classic_sta_lta4_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    Moving_average_700_mean_pr = make_agg_primitive(function = Moving_average_700_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    Moving_average_1500_mean_pr = make_agg_primitive(function = Moving_average_1500_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    Moving_average_3000_mean_pr = make_agg_primitive(function = Moving_average_3000_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    Moving_average_6000_mean_pr = make_agg_primitive(function = Moving_average_6000_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    exp_Moving_average_300_mean_pr = make_agg_primitive(function = exp_Moving_average_300_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    exp_Moving_average_3000_mean_pr = make_agg_primitive(function = exp_Moving_average_3000_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    exp_Moving_average_30000_mean_pr = make_agg_primitive(function = exp_Moving_average_30000_mean,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    iqr_pr = make_agg_primitive(function = iqr,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q999_pr = make_agg_primitive(function = q999,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q001 = make_agg_primitive(function = q001,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    ave10_pr = make_agg_primitive(function = ave10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    ave_roll_std_10_pr = make_agg_primitive(function = ave_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_roll_std_10_pr = make_agg_primitive(function = std_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_roll_std_10_pr = make_agg_primitive(function = max_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_roll_std_10_pr = make_agg_primitive(function = min_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q01_roll_std_10_pr = make_agg_primitive(function = q01_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q05_roll_std_10_pr = make_agg_primitive(function = q05_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q95_roll_std_10_pr = make_agg_primitive(function = q95_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q99_roll_std_10_pr = make_agg_primitive(function = q99_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_abs_roll_std_10_pr = make_agg_primitive(function = av_change_abs_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_rate_roll_std_10_pr = make_agg_primitive(function = av_change_rate_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_max_roll_std_10_pr = make_agg_primitive(function = abs_max_roll_std_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_roll_mean_10_pr = make_agg_primitive(function = std_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_roll_mean_10_pr = make_agg_primitive(function = max_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_roll_mean_10_pr = make_agg_primitive(function = min_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q01_roll_mean_10_pr = make_agg_primitive(function = q01_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q05_roll_mean_10_pr = make_agg_primitive(function = q05_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q95_roll_mean_10_pr = make_agg_primitive(function = q95_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q99_roll_mean_10_pr = make_agg_primitive(function = q99_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_abs_roll_mean_10_pr = make_agg_primitive(function = av_change_abs_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_rate_roll_mean_10_pr = make_agg_primitive(function = av_change_rate_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_max_roll_mean_10_pr = make_agg_primitive(function = abs_max_roll_mean_10,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    ave_roll_std_100_pr = make_agg_primitive(function = ave_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_roll_std_100_pr = make_agg_primitive(function = std_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_roll_std_100_pr = make_agg_primitive(function = max_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_roll_std_100_pr = make_agg_primitive(function = min_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q01_roll_std_100_pr = make_agg_primitive(function = q01_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q05_roll_std_100_pr = make_agg_primitive(function = q05_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q95_roll_std_100_pr = make_agg_primitive(function = q95_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q99_roll_std_100_pr = make_agg_primitive(function = q99_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_abs_roll_std_100_pr = make_agg_primitive(function = av_change_abs_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_rate_roll_std_100_pr = make_agg_primitive(function = av_change_rate_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_max_roll_std_100_pr = make_agg_primitive(function = abs_max_roll_std_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_roll_mean_100_pr = make_agg_primitive(function = std_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_roll_mean_100_pr = make_agg_primitive(function = max_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_roll_mean_100_pr = make_agg_primitive(function = min_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q01_roll_mean_100_pr = make_agg_primitive(function = q01_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q05_roll_mean_100_pr = make_agg_primitive(function = q05_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q95_roll_mean_100_pr = make_agg_primitive(function = q95_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q99_roll_mean_100_pr = make_agg_primitive(function = q99_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_abs_roll_mean_100_pr = make_agg_primitive(function = av_change_abs_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_rate_roll_mean_100_pr = make_agg_primitive(function = av_change_rate_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_max_roll_mean_100_pr = make_agg_primitive(function = abs_max_roll_mean_100,
                              input_types = [Numeric],
                              return_type = Numeric)

    ave_roll_std_1000_pr = make_agg_primitive(function = ave_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_roll_std_1000_pr = make_agg_primitive(function = std_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_roll_std_1000_pr = make_agg_primitive(function = max_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_roll_std_1000_pr = make_agg_primitive(function = min_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q01_roll_std_1000_pr = make_agg_primitive(function = q01_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q05_roll_std_1000_pr = make_agg_primitive(function = q05_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q95_roll_std_1000_pr = make_agg_primitive(function = q95_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q99_roll_std_1000_pr = make_agg_primitive(function = q99_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_abs_roll_std_1000_pr = make_agg_primitive(function = av_change_abs_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_rate_roll_std_1000_pr = make_agg_primitive(function = av_change_rate_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_max_roll_std_1000_pr = make_agg_primitive(function = abs_max_roll_std_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_roll_mean_1000_pr = make_agg_primitive(function = std_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_roll_mean_1000_pr = make_agg_primitive(function = max_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_roll_mean_1000_pr = make_agg_primitive(function = min_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q01_roll_mean_1000_pr = make_agg_primitive(function = q01_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q05_roll_mean_1000_pr = make_agg_primitive(function = q05_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q95_roll_mean_1000_pr = make_agg_primitive(function = q95_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q99_roll_mean_1000_pr = make_agg_primitive(function = q99_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_abs_roll_mean_1000_pr = make_agg_primitive(function = av_change_abs_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_rate_roll_mean_1000_pr = make_agg_primitive(function = av_change_rate_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_max_roll_mean_1000_pr = make_agg_primitive(function = abs_max_roll_mean_1000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    ave_roll_std_10000_pr = make_agg_primitive(function = ave_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_roll_std_10000_pr = make_agg_primitive(function = std_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_roll_std_10000_pr = make_agg_primitive(function = max_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_roll_std_10000_pr = make_agg_primitive(function = min_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q01_roll_std_10000_pr = make_agg_primitive(function = q01_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q05_roll_std_10000_pr = make_agg_primitive(function = q05_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q95_roll_std_10000_pr = make_agg_primitive(function = q95_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q99_roll_std_10000_pr = make_agg_primitive(function = q99_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_abs_roll_std_10000_pr = make_agg_primitive(function = av_change_abs_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_rate_roll_std_10000_pr = make_agg_primitive(function = av_change_rate_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_max_roll_std_10000_pr = make_agg_primitive(function = abs_max_roll_std_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    std_roll_mean_10000_pr = make_agg_primitive(function = std_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    max_roll_mean_10000_pr = make_agg_primitive(function = max_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    min_roll_mean_10000_pr = make_agg_primitive(function = min_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q01_roll_mean_10000_pr = make_agg_primitive(function = q01_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q05_roll_mean_10000_pr = make_agg_primitive(function = q05_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q95_roll_mean_10000_pr = make_agg_primitive(function = q95_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    q99_roll_mean_10000_pr = make_agg_primitive(function = q99_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_abs_roll_mean_10000_pr = make_agg_primitive(function = av_change_abs_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    av_change_rate_roll_mean_10000_pr = make_agg_primitive(function = av_change_rate_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)
    
    abs_max_roll_mean_10000_pr = make_agg_primitive(function = abs_max_roll_mean_10000,
                              input_types = [Numeric],
                              return_type = Numeric)

    return gmean_pr, hmean_pr, kstatvar1_pr, kstat2_pr, kstatvar2_pr, kstat3_pr, kstat4_pr, \
            avg_change_pr, avg_change_rate_pr, range_pr, std_first_50000_pr, \
            std_last_50000_pr, std_first_10000_pr, std_last_10000_pr, avg_first_50000_pr, \
            avg_last_50000_pr, avg_first_10000_pr, avg_last_10000_pr, min_first_50000_pr, \
            min_last_50000_pr, min_first_10000_pr, min_last_10000_pr, max_first_50000_pr, \
            max_last_50000_pr, max_first_10000_pr, max_last_10000_pr, max_to_min_pr, \
            count_big_pr, sum_func_pr, avg_change_rate_first_50000_pr, avg_change_rate_last_50000_pr, \
            avg_change_rate_first_10000_pr, avg_change_rate_last_10000_pr, q95_pr, \
            q99_pr, q05_pr, q01_pr, abs_q95_pr, abs_q99_pr, trend_pr, abs_trend_pr, \
            abs_mean_pr, abs_std_pr, mad_pr, kurt_pr, skew_pr, med_pr, Hilbert_mean_pr, \
            Hann_window_mean_pr, classic_sta_lta1_mean_pr, classic_sta_lta2_mean_pr, \
            classic_sta_lta3_mean_pr, classic_sta_lta4_mean_pr, Moving_average_700_mean_pr, \
            Moving_average_1500_mean_pr, Moving_average_3000_mean_pr, Moving_average_6000_mean_pr, \
            exp_Moving_average_300_mean_pr, exp_Moving_average_3000_mean_pr, \
            exp_Moving_average_30000_mean_pr, iqr_pr, q999_pr, q001, ave10_pr, \
            ave_roll_std_10_pr, std_roll_std_10_pr, max_roll_std_10_pr, min_roll_std_10_pr, \
            q01_roll_std_10_pr, q05_roll_std_10_pr, q95_roll_std_10_pr, q99_roll_std_10_pr, \
            av_change_abs_roll_std_10_pr, av_change_rate_roll_std_10_pr, abs_max_roll_std_10_pr, \
            std_roll_mean_10_pr, max_roll_mean_10_pr, min_roll_mean_10_pr, q01_roll_mean_10_pr, \
            q05_roll_mean_10_pr, q95_roll_mean_10_pr, q99_roll_mean_10_pr, \
            av_change_abs_roll_mean_10_pr, av_change_rate_roll_mean_10_pr, \
            abs_max_roll_mean_10_pr, ave_roll_std_100_pr, std_roll_std_100_pr, \
            max_roll_std_100_pr, min_roll_std_100_pr, q01_roll_std_100_pr, \
            q05_roll_std_100_pr, q95_roll_std_100_pr, q99_roll_std_100_pr, \
            av_change_abs_roll_std_100_pr, av_change_rate_roll_std_100_pr, \
            abs_max_roll_std_100_pr, std_roll_mean_100_pr, max_roll_mean_100_pr, \
            min_roll_mean_100_pr, q01_roll_mean_100_pr, q05_roll_mean_100_pr, \
            q95_roll_mean_100_pr, q99_roll_mean_100_pr, av_change_abs_roll_mean_100_pr, \
            av_change_rate_roll_mean_100_pr, abs_max_roll_mean_100_pr, ave_roll_std_1000_pr, \
            std_roll_std_1000_pr, max_roll_std_1000_pr, min_roll_std_1000_pr, \
            q01_roll_std_1000_pr, q05_roll_std_1000_pr, q95_roll_std_1000_pr, \
            q99_roll_std_1000_pr, av_change_abs_roll_std_1000_pr, \
            av_change_rate_roll_std_1000_pr, abs_max_roll_std_1000_pr, \
            std_roll_mean_1000_pr, max_roll_mean_1000_pr, min_roll_mean_1000_pr, \
            q01_roll_mean_1000_pr, q05_roll_mean_1000_pr, q95_roll_mean_1000_pr, \
            q99_roll_mean_1000_pr, av_change_abs_roll_mean_1000_pr, \
            av_change_rate_roll_mean_1000_pr, abs_max_roll_mean_1000_pr, \
            ave_roll_std_10000_pr, std_roll_std_10000_pr, max_roll_std_10000_pr, \
            min_roll_std_10000_pr, q01_roll_std_10000_pr, q05_roll_std_10000_pr, \
            q95_roll_std_10000_pr, q99_roll_std_10000_pr, av_change_abs_roll_std_10000_pr, \
            av_change_rate_roll_std_10000_pr, abs_max_roll_std_10000_pr, \
            std_roll_mean_10000_pr, max_roll_mean_10000_pr, min_roll_mean_10000_pr, \
            q01_roll_mean_10000_pr, q05_roll_mean_10000_pr, q95_roll_mean_10000_pr, \
            q99_roll_mean_10000_pr, av_change_abs_roll_mean_10000_pr, \
            av_change_rate_roll_mean_10000_pr, abs_max_roll_mean_10000_pr
