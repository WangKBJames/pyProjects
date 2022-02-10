# -*- coding: utf-8 -*-
"""
Aim：两变量线形拟合
"""
import numpy as np
# from scipy import signal
from scipy.optimize import leastsq
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# hypothesis function
def hypothesis_func(w, x):
    '''

    :param w:
    :param x:
    :return:
    '''
    w1, w0 = w
    return w1 * x + w0


# error function
def error_func(w, train_x, train_y):
    '''

    :param w:
    :param train_x:
    :param train_y:
    :return:
    '''
    return hypothesis_func(w, train_x) - train_y


def dump_fit_func(w_fit):
    '''

    :param w_fit:
    :return:
    '''
    w1, w0 = w_fit
    print("fitting line=", str(w1) + "*x + " + str(w0))
    return


# square error平方差函数
def dump_fit_cost(w_fit, train_x, train_y):
    '''

    :param w_fit:
    :param train_x:
    :param train_y:
    :return:
    '''
    error = error_func(w_fit, train_x, train_y, "")
    square_error = sum(e * e for e in error)
    print('fitting cost:', str(square_error))
    return square_error

'''

计算相关函数
输入：
x_signal：float[]，时程数据1
y_signal：float[]，时程数据2
输出：
x_fit: float[]，频率，单位：Hz
y_fit：float[]，相干函数
a: float，相关函y=a*x+b,系数a
b: float，相关函y=a*x+b,a
corr：float，相关系数

'''


def correlation(x_signal, y_signal):
    '''

    :param x_signal: float[]，时程数据1
    :param y_signal: float[]，时程数据2
    :return:
    x_fit: float[]，频率，单位：Hz
    y_fit：float[]，相干函数
    a: float，相关函数y=a*x+b,系数a
    b: float，相关函数y=a*x+b,系数b
    corr：float，相关系数
    '''
    if type(x_signal) is not np.ndarray:
        x_signal = np.array(x_signal, dtype='float')
    if type(x_signal) is not np.ndarray:
        y_signal = np.array(y_signal, dtype='float')
    w_init = [1.0, 1.0]
    fit_ret = leastsq(error_func, w_init, args=(x_signal, y_signal))
    a, b = fit_ret[0]
    x_fit = x_signal
    y_fit = hypothesis_func([a, b], x_fit)
    corr = pearsonr(x_signal, y_signal)
    # return x_fit.tolist(), y_fit.tolist(), np.float(a), np.float(b), np.float(corr[0])
    return x_fit.tolist(), y_fit.tolist(), float(a), float(b), float(corr[0])


if __name__ == "__main__":
    # train set
    x_signal = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
    y_signal = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])*0.01
    x_fit, y_fit, a, b, corr = correlation(x_signal, y_signal)

    # show result by figure
    # plt.figure(1)
    plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    plt.title('linear regression by scipy leastsq')
    plt.scatter(x_signal, y_signal, color='b', label='train set')
    plt.plot(x_fit, y_fit, color='r', label='fitting line')
    plt.legend(loc='lower right')  # label面板放到figure的右下角
    plt.show()