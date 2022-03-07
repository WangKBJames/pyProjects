# -*- coding: utf-8 -*-
"""
Aim：频谱图、累积量(速度)提取
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.stats import pearsonr
import math

main_path = r".\feature_extraction\\"

def spectrum(y_signal, fs):
    """

    :param y_signal:
    :param fs:
    :return:
    """
    if type(y_signal) is not np.ndarray:
        y_signal = np.array(y_signal, dtype='float')
    y_signal[np.isnan(y_signal)] = np.nanmean(y_signal)
    fft_y = np.fft.rfft(y_signal)
    n = int(len(y_signal))
    abs_fy = np.abs(fft_y) / n * 2
    freqs = np.linspace(0, fs / 2, int(n / 2 + 1))
    frqp = freqs[np.argmax(abs_fy)]
    return [freqs.tolist(), abs_fy.tolist(), float(frqp)]


# hypothesis function
def hypothesis_func(w, x):
    w1, w0 = w
    return w1 * x + w0


# error function
def error_func(w, train_x, train_y):
    return hypothesis_func(w, train_x) - train_y


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


def trend(y_signal, fs):
    '''
    :param y_signal:
    :param fs:
    :return:
    '''
    if type(y_signal) is not np.ndarray:
        y_signal = np.array(y_signal, dtype='float')
    y_signal = movmean(y_signal, int(60*fs))
    dt = 1 / float(fs)
    t = np.arange(start=0.0, stop=len(y_signal) * dt, step=dt, dtype='float')
    w_init = [1.0, 1.0]
    fit_ret = leastsq(error_func, w_init, args=(t, y_signal))
    a, b = fit_ret[0]
    y_fit = hypothesis_func([a, b], t)
    corr = pearsonr(y_fit, y_signal)
    return [y_signal, y_fit.tolist(), float(a), float(b), float(corr[0])]



def cumulation(y_signal, fs):
    '''
    :param y_signal:
    :param fs:
    :return:
    '''
    if type(y_signal) is not np.ndarray:
        y_signal = np.array(y_signal, dtype='float')
    y_cum = np.array([0], dtype='float') + np.cumsum(np.abs(np.diff(y_signal)))
    dt = 1 / float(fs)
    t = np.arange(start=0.0, stop=len(y_cum) * dt, step=dt, dtype='float')
    w_init = [1.0, 1.0]
    fit_ret = leastsq(error_func, w_init, args=(t, y_cum))
    a, b = fit_ret[0]
    y_fit = hypothesis_func([a, b], t)
    corr = pearsonr(y_fit, y_cum)
    return [y_cum.tolist(), y_fit.tolist(), float(a), float(b), float(corr[0])]


def process():
    par = np.loadtxt(main_path + r"input\par.txt", dtype='float')
    type_num = par[0]
    fs = par[1]
    data_1 = np.loadtxt(main_path + r"input\shuju1.txt", dtype='float')
    if type_num == 1:
        y_movmean, y_fit, a, b, corr = trend(data_1, fs)
        np.savetxt(main_path + r"output\fig2_y_1.txt", y_movmean)
        np.savetxt(main_path + r"output\fig2_y_2.txt", y_fit)
        doc_str = "趋势斜率：{0:.3g}\n" \
                  "相关系数：{1:.3f}".format(a, corr)
    elif type_num == 2:
        freqs, abs_fy, frqp = spectrum(data_1, fs)
        np.savetxt(main_path + r"output\fig2_x_1.txt", freqs)
        np.savetxt(main_path + r"output\fig2_y_1.txt", abs_fy)
        doc_str = "峰值频率：{0:.3g}Hz".format(frqp)
    elif type_num == 3:
        y_cum, y_fit, a, b, corr = cumulation(data_1, fs)
        np.savetxt(main_path + r"output\fig2_y_1.txt", y_cum)
        np.savetxt(main_path + r"output\fig2_y_2.txt", y_fit)
        doc_str = "趋势斜率：{0:.3g}\n" \
                  "相关系数：{1:.3f}".format(a, corr)
    with open(main_path + r"output\shuoming.txt", "w") as f:
        f.write(doc_str)
    return


if __name__ == "__main__":
    # train set
    # y_signal = np.arange(10000) + np.random.randn(10000)
    # fs = 1.0
    # f, fy, frqp = spectrum(y_signal, fs)
    # plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    # plt.title('frequency spectrum')
    # # plt.scatter(x_signal, y_signal, color='b', label='train set')
    # plt.plot(f, fy, color='r', label='spectrum')
    # plt.legend(loc='lower right')  # label面板放到figure的右下角
    # plt.show()

    # y_signal = np.random.random(10000)-0.5
    # fs = 1
    # y_cum, y_fit, a, b, corr = cumulation(y_signal, fs)
    # plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    # plt.title('cumsum')
    # # plt.scatter(x_signal, y_signal, color='b', label='train set')
    # plt.plot(y_cum, color='r', label='cumsum')
    # plt.plot(y_fit, color='b', label='cumfit')
    # plt.legend(loc='lower right')  # label面板放到figure的右下角
    # plt.show()

    # np.savetxt(main_path + r"input\shuju1.txt", y_signal)
    process()
