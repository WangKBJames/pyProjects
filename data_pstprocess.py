# -*- coding: utf-8 -*-
"""
Aim：据极值、分位数、均值（k线图）、均方根（折线图）
"""
import numpy as np
# from scipy import signal
from scipy.optimize import leastsq
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.interpolate as spi
import math


def movmean(y_signal, win_n):
    '''
    :param y_signal:
    :param win_n:
    :return:
    '''
    # conv = np.ones(int(win_n), dtype='float')
    y_mean = np.array([])
    for i in range(0, len(y_signal)):
        if i < int(win_n // 2):
            y_mean = np.append(y_mean, np.nanmean(y_signal[:i + int(math.ceil(int(win_n) / 2))]))
        elif i >= len(y_signal) - int(math.floor(win_n / 2)):
            y_mean = np.append(y_mean, np.nanmean(y_signal[i - int(math.floor(win_n / 2)):]))
        else:
            y_mean = np.append(y_mean, np.nanmean(y_signal[i - math.floor(win_n / 2):i + int(math.ceil(win_n / 2))]))
    return y_mean.tolist()


def movstd(y_signal, win_n):
    '''

    :param y_signal:
    :param win_n:
    :return:
    '''
    y_std = np.array([])
    for i in range(0, len(y_signal)):
        if i < int(win_n // 2):
            y_std = np.append(y_std, np.nanstd(y_signal[:i + int(math.ceil(int(win_n) / 2))]))
        elif i >= len(y_signal) - int(math.floor(win_n / 2)):
            y_std = np.append(y_std, np.nanstd(y_signal[i - int(math.floor(win_n / 2)):]))
        else:
            y_std = np.append(y_std, np.nanstd(y_signal[i - math.floor(win_n / 2):i + int(math.ceil(win_n / 2))]))
    return y_std.tolist()


def extreme(ydata,fs):
    '''
    复选框选择”极值时“
    :param ydata: float[] 输入特征数据(图1的y轴，x轴为时间）
    :param fs: float 采用频率
    :return:list[][]  十分钟极值（3条折线）
    list[0]:  y_max : float[], 十分钟最大值序列，图2 y轴，x轴为时间
    list[1]:  y_min: float[], 十分钟最小值序列， 图2 y轴，x轴为时间
    list[2]:  y_mean: float[], 均值序列， 图2 y轴，x轴为时间
    '''

    pass


def data_quantile(ydata,fs):
    '''
    复选框选择“分位数”时
    :param ydata: float[] 输入特征数据
    :param fs: float 采用频率
    :return: list[][]
    list[0]: 0.75 分位数序列，list[0]:  y_75: float[],图2 y轴，x轴为时间
    list[1]: 0.5 分位数序列，list[0]:  y_75: float[],图2 y轴，x轴为时间
    list[2]: 0.25 分位数序列，list[1]: y_25: float[],图2 y轴，x轴为时间
    '''
    pass

def data_std(ydata,fs):
    '''
    复选框选择“均方根”时
    :param ydata: float[] 输入特征数据
    :param fs: float 采用频率
    :return:
    均方根， float[]，图2 y轴，x轴为时间
    '''
    pass

def data_untrend(ydata,fs):
    '''
    复选框选择“去趋势”时
    :param ydata: float[] 输入特征数据
    :param fs: float 采用频率
    :return:
    去趋势数据序列， float[]，图2 y轴，x轴为时间
    '''
    pass




def get_range(x, data_x, frac):
    '''
    以x为中心，找着frac的比例截取数据，端部数据取同等长度
    :param x: 数据中心
    :param data_x: 数据x
    :param frac: 截取比例
    :return: np.array
    '''
    x_ind = np.argwhere(data_x == x)[0][0]
    if frac >= 1:
        half_len_w = frac//2
    else:
        half_len_w = int(np.floor((data_x.shape[0] * frac) // 2))
    len_x_list = 2 * half_len_w + 1
    if (x_ind - half_len_w) < 0:
        x_list = data_x[0:len_x_list]
        x_loc = x_ind
    elif (x_ind + half_len_w) >= len(data_x):
        x_list = data_x[-len_x_list:]
        x_loc = x_ind - len(data_x)
    else:
        x_list = data_x[x_ind - half_len_w:x_ind + half_len_w + 1]
        x_loc = half_len_w
    return x_list, x_loc


def calFuncW(x_list, x_loc):
    '''
    以w函数计算权值函数
    :param x_list: 计算权值的x序列
    :return:
    '''
    len_x_list = len(x_list)
    if 2 * x_loc + 1 == len_x_list:
        x_norm = np.linspace(-1, 1, len_x_list + 2)
        w = (1 - x_norm ** 2) ** 2
        w = w[1:-1]
    elif x_loc >= 0:
        x_norm = np.linspace(-1, 1, (len_x_list - x_loc - 1) * 2 + 3)
        w = (1 - x_norm ** 2) ** 2
        w = w[-len_x_list - 1:-1]
    else:
        x_norm = np.linspace(-1, 1, (len_x_list + x_loc) * 2 + 3)
        w = (1 - x_norm ** 2) ** 2
        w = w[1:len_x_list + 1]
    return w


def weightRegression(x_list, y_list, w, fitfunc="T"):
    '''
    权重回归分析
    :param x_list:
    :param y_list:
    :param w:
    :return:
    '''
    # x2 = x_list.reshape(1, len(x_list))
    y2 = y_list.reshape(1, len(x_list))
    w2 = w.reshape(1, len(x_list))
    # y_list_regress = x2.dot(np.linalg.inv(x2.T.dot((w * x_list).reshape(1, len(x_list))))).dot(x2.T.dot((w * y_list).reshape(1, len(x_list))))
    if fitfunc == "B":
        x2 = np.ones([2, len(x_list)])
        x2[0] = x_list
        y_list_regress = x2.T.dot(np.linalg.inv(x2.dot((w2 * x2).T))).dot(x2.dot((w2 * y2).T))
    elif fitfunc == "T":
        x2 = np.ones([3, len(x_list)])
        x2[0] = x_list ** 2
        x2[1] = x_list
        y_list_regress = x2.T.dot(np.linalg.inv(x2.dot((w2 * x2).T))).dot(x2.dot((w2 * y2).T))
    return y_list_regress.reshape(len(y_list))


def cal_new_weight(y_hat, y_list, w, wfunc="B"):
    '''
    计算局部回归调整后权重
    :param y_hat: 局部回归后输出数据
    :param data_y: 原始数据
    :param func: string, "B"二次权重函数，"w"三次权重函数
    :return:
    '''
    err = y_list - y_hat
    s = np.nanmedian(np.abs(err))
    err_norm = err / 6 / s
    if wfunc == "B":
        delta_k = (1 - err_norm ** 2) ** 2
    elif wfunc == "W":
        delta_k = (1 - np.abs(err_norm) ** 3) ** 3
    delta_k[abs(err_norm) > 1] = 0
    new_w = delta_k * w
    return new_w


def rlowess(data_x, data_y, frac, iters=2):
    '''
    鲁棒性的加权回归：
    Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”. Journal of the American Statistical Association 74 (368): 829-836.
    :param data_x:
    :param data_y:
    :param frac:
    :return:
    '''
    data_y_hat = np.ones_like(data_y)
    half_len_w = int(np.floor((data_x.shape[0] * frac) // 2))
    for x in data_x:
        x_list, x_loc = get_range(x, data_x, frac)
        new_w = calFuncW(x_list, x_loc)
        y_hat = weightRegression(x_list, data_y[x_list], new_w)
        for it in range(iters):
            new_w = cal_new_weight(y_hat, data_y[x_list], new_w, wfunc="B")
            y_hat = weightRegression(x_list, data_y[x_list], new_w, "B")
        data_y_hat[x] = y_hat[x_loc]
    return data_y_hat


def rloess(data_x, data_y, frac, step=1, iters=2):
    '''
    鲁棒性的加权回归：
    Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”. Journal of the American Statistical Association 74 (368): 829-836.
    :param data_x:
    :param data_y:
    :param frac:
    :param step:
    :param iters:
    :return:
    '''
    # data_y_hat = np.ones_like(data_y)
    if frac >= 1:
        half_len_w = frac//2
    else:
        half_len_w = int(np.floor((data_x.shape[0] * frac) // 2))
    data_x_step = data_x[0::step]
    if data_x_step[-1] != data_x[-1]:
        data_x_step = np.append(data_x_step, data_x[-1])
    data_y_hat_step = np.random.random(len(data_x_step))
    w_list = np.random.random(len(data_x_step))
    for x in range(len(data_x_step)):
        x_list, x_loc = get_range(data_x_step[x], data_x, frac)
        new_w = calFuncW(x_list, x_loc)
        y_hat = weightRegression(x_list, data_y[x_list], new_w)
        for it in range(iters):
            new_w = cal_new_weight(y_hat, data_y[x_list], new_w, wfunc="B")
            y_hat = weightRegression(x_list, data_y[x_list], new_w)
        data_y_hat_step[x] = y_hat[x_loc]
        w_list[x] = new_w[x_loc]
    data_y_hat_rep = spi.splrep(data_x_step, data_y_hat_step, k=2)
    data_y_hat = spi.splev(data_x, data_y_hat_rep)
    return data_y_hat, w_list