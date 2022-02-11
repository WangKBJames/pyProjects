# -*- coding: utf-8 -*-
"""
Aim：频谱图、累积量(速度)提取
"""
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.stats import pearsonr


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
    return freqs.tolist(), abs_fy.tolist(), float(frqp)


# hypothesis function
def hypothesis_func(w, x):
    w1, w0 = w
    return w1 * x + w0


# error function
def error_func(w, train_x, train_y):
    return hypothesis_func(w, train_x) - train_y


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
    return y_cum.tolist(), y_fit.tolist(), float(a), float(b), float(corr[0])


if __name__ == "__main__":
    # train set
    y_signal = np.array([-1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2, -1.2, 1.2]) * 0.01
    fs = 1.0
    f, fy, frqp = spectrum(y_signal, fs)
    plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    plt.title('frequency spectrum')
    # plt.scatter(x_signal, y_signal, color='b', label='train set')
    plt.plot(f, fy, color='r', label='spectrum')
    plt.legend(loc='lower right')  # label面板放到figure的右下角
    plt.show()

    y_signal = np.random.random(30)-0.5
    fs = 1
    y_cum, y_fit, a, b, corr = cumulation(y_signal, fs)
    plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    plt.title('cumsum')
    # plt.scatter(x_signal, y_signal, color='b', label='train set')
    plt.plot(y_cum, color='r', label='cumsum')
    plt.plot(y_fit, color='b', label='cumfit')
    plt.legend(loc='lower right')  # label面板放到figure的右下角
    plt.show()
