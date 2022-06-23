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
from scipy.fft import rfft, irfft

main_path = r".\correlation\\"


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


def data_unnan(data):
    '''
    线形差值去掉nan
    :param data:
    :return:
    '''
    nan_ind = np.argwhere(np.isnan(data))
    x = np.arange(len(data))
    x = np.delete(x, nan_ind)
    y = np.delete(data, nan_ind)
    y_fit = np.interp(nan_ind, x, y)
    data[nan_ind] = y_fit
    return data


def data_align(x, y, long_fit=True):
    '''
    不同长度数据对齐为相同长度
    :param x:
    :param y:
    :return:
    '''
    if type(x) is not np.ndarray:
        x = np.array(x, dtype='float')
    if type(y) is not np.ndarray:
        y = np.array(y, dtype='float')

    if np.isnan(x).any():
        x = data_unnan(x)
    if np.isnan(y).any():
        y = data_unnan(y)
    if len(x) > len(y):
        r = np.round(len(x) / len(y))
        if r == 1:
            x_align = x[:len(y)]
            y_align = y
        else:
            if long_fit:
                tx = np.arange(len(x))
                ty = np.arange(0, r * len(y), r)
                y_align = np.interp(tx, ty, y)
                x_align = x
            elif not long_fit:
                x_align = x[np.linspace(r // 2, r // 2 + (len(y) - 1) * r, len(y))]
                y_align = y
    elif len(x) < len(y):
        r = np.round(len(y) / len(x))
        if r == 1:
            y_align = y[:len(x)]
            x_align = x
        else:
            if long_fit:
                ty = np.arange(len(y))
                tx = np.arange(0, r * len(x), r)
                x_align = np.interp(ty, tx, x)
                y_align = y
            elif not long_fit:
                y_align = y[np.linspace(r // 2, r // 2 + (len(x) - 1) * r, len(x))]
                x_align = x
    else:
        x_align = x
        y_align = y
    return x_align, y_align


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
    x_signal[np.isnan(x_signal)] = np.nanmean(x_signal)
    if type(y_signal) is not np.ndarray:
        y_signal = np.array(y_signal, dtype='float')
    y_signal[np.isnan(y_signal)] = np.nanmean(y_signal)
    if len(x_signal) != len(y_signal):
        if len(x_signal) > len(y_signal):
            x_signal = x_signal[0:len(y_signal)]
        else:
            y_signal = y_signal[0:len(x_signal)]
    w_init = [1.0, 1.0]
    fit_ret = leastsq(error_func, w_init, args=(x_signal, y_signal))
    a, b = fit_ret[0]
    x_fit = x_signal
    y_fit = hypothesis_func([a, b], x_fit)
    ind_x_fit = np.argsort(x_fit)
    x_fit = np.sort(x_fit)
    y_fit = y_fit[ind_x_fit]
    corr = pearsonr(x_signal, y_signal)
    # return x_fit.tolist(), y_fit.tolist(), np.float(a), np.float(b), np.float(corr[0])
    return [x_fit.tolist(), y_fit.tolist(), float(a), float(b), float(corr[0])]


def xcorr(x_signal, y_signal):
    return irfft(rfft(x_signal) * rfft(y_signal[::-1]))


def max_corr_data(x_signal, y_signal):
    xy_corr = xcorr(x_signal, y_signal)
    max_ind = np.argmax(xy_corr)


def process():
    data_1 = np.loadtxt(main_path + r"input\shuju1.txt", dtype='float')
    data_2 = np.loadtxt(main_path + r"input\shuju2.txt", dtype='float')
    data_1_align, data_2_align = data_align(data_1, data_2, long_fit=True)
    x_fit, y_fit, a, b, corr = correlation(data_1_align, data_2_align)
    ind_data_1 = np.argsort(data_1_align)
    data_1_align = data_1_align[ind_data_1]
    data_2_align = data_2_align[ind_data_1]
    np.savetxt(main_path + r"output\fig2_x_1.txt", data_1_align)
    np.savetxt(main_path + r"output\fig2_y_1.txt", data_2_align)
    np.savetxt(main_path + r"output\fig2_x_2.txt", x_fit)
    np.savetxt(main_path + r"output\fig2_y_2.txt", y_fit)
    doc_str = "函数形式：{0:.3g}x+{1:.3g}\n" \
              "相关系数：{2:.3f}".format(a, b, np.abs(corr))
    with open(main_path + r"output\shuoming.txt", "w", encoding="utf-8") as f:
        f.write(doc_str)
    return


if __name__ == "__main__":
    # from dataReader import bin_data

    if False:
        x_signal = np.array([8.19, np.nan, 6.39, 8.71, 4.7, 2.66, 3.78])
        y_signal = np.array([7.01, 2.78, 6.47, 6.71, 4.1, np.nan, 4.05]) * 0.01
        x_fit, y_fit, a, b, corr = correlation(x_signal, y_signal)

        # show result by figure
        # plt.figure(1)
        plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
        plt.title('linear regression by scipy leastsq')
        plt.scatter(x_signal, y_signal, color='b', label='train set')
        plt.plot(x_fit, y_fit, color='r', label='fitting line')
        plt.legend(loc='lower right')  # label面板放到figure的右下角
        plt.show()

        np.savetxt(main_path + r"input\shuju1.txt", x_signal)
        np.savetxt(main_path + r"input\shuju2.txt", y_signal)
    if False:
        x = np.random.random(1024)
        y = np.random.random(1024)
        corrxy = xcorr(x, y)
        plt.plot(corrxy)
        plt.show()
        np.arange()
    if False:
        jw_path = r"Z:\江苏控股桥群\江阴\JW"
        wy_path = r"Z:\江苏控股桥群\江阴\WY"
        jw_sensor = "JW020101"
        wy_sensor = "WY020101"
        t_start_list = [2021, 4, 1, 0, 0, 0]
        t_end_list = [2021, 4, 10, 0, 0, 0]
        _, wy_data = bin_data(wy_path, wy_sensor, t_start_list, t_end_list, sample_frq=1)
        _, jw_data = bin_data(jw_path, jw_sensor, t_start_list, t_end_list, sample_frq=1 / 300)
        np.savetxt(main_path + r"input\shuju1.txt", wy_data)
        np.savetxt(main_path + r"input\shuju2.txt", jw_data)
        process()


        # wy, jw = data_align(wy_data, jw_data, long_fit=True)
        # x_fit, y_fit, a, b, corr = correlation(jw, wy)
        # fig = plt.figure(figsize=(12, 8))  # 定义图并设置画板尺寸
        # fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        # ax1 = fig.add_subplot(311)  # 定义子图
        # ax1.plot(wy, 'r')
        # ax2 = fig.add_subplot(312)
        # ax2.plot(jw, 'b')
        # ax3 = fig.add_subplot(313)
        # ax3.scatter(jw, wy)
        # ax3.plot(x_fit, y_fit, 'r')
        # plt.show()
    if True:
        process()
        data_11 = np.loadtxt(main_path + r"input\shuju1.txt", dtype='float')
        data_12 = np.loadtxt(main_path + r"input\shuju2.txt", dtype='float')
        print(np.min(data_11))
        print(np.min(data_12))
        plt.plot(data_11, color='r', label='fitting line')
        plt.show()
        # data_21 = np.loadtxt(main_path + r"output\fig2_x_2.txt", dtype='float')
        # data_22 = np.loadtxt(main_path + r"output\fig2_y_2.txt", dtype='float')
        # data_11 = np.loadtxt(main_path + r"output\fig2_x_1.txt", dtype='float')
        # data_12 = np.loadtxt(main_path + r"output\fig2_y_1.txt", dtype='float')
        # data_21 = np.loadtxt(main_path + r"output\fig2_x_2.txt", dtype='float')
        # data_22 = np.loadtxt(main_path + r"output\fig2_y_2.txt", dtype='float')
        # plt.scatter(data_11, data_12, color='b', label='train set')
        # plt.plot(data_21, data_22, color='r', label='fitting line')
        # plt.legend(loc='lower right')  # label面板放到figure的右下角
        # plt.show()
