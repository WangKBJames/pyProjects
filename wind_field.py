# -*- coding: utf-8 -*-

import numpy as np
import math
# import pandas as pd
# from scipy.fftpack import fft
# from scipy.optimize import leastsq
import matplotlib.pyplot as plt


# from scipy.optimize import leastsq
# from scipy.stats import pearsonr


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
            y_mean = np.append(y_mean, np.mean(y_signal[:i + int(math.ceil(int(win_n) / 2))]))
        elif i >= len(y_signal) - int(math.floor(win_n / 2)):
            y_mean = np.append(y_mean, np.mean(y_signal[i - int(math.floor(win_n / 2)):]))
        else:
            y_mean = np.append(y_mean, np.mean(y_signal[i - math.floor(win_n / 2):i + int(math.ceil(win_n / 2))]))
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


def wind_stats(wind_velocity, sample_frq=1):
    '''
    :param wind_velocity:
    :param horizontal_angle:
    :param attack_angle:
    :param sample_frq:
    :return:
    '''
    wmean_2min = movmean(wind_velocity, int(120 / sample_frq))
    wmean_10min = movmean(wind_velocity, int(600 / sample_frq))
    wpulse = wind_velocity - wmean_10min
    wstd_10min = movstd(wind_velocity, int(600 / sample_frq))
    return wmean_2min, wmean_10min, wpulse, wstd_10min


def attack_angle(angle):
    '''

    :param angle:
    :return:
    '''
    return angle


def spectrum(y_signal, fs):
    """

    :param y_signal:
    :param fs:
    :return:
    """
    if type(y_signal) is not np.ndarray:
        y_signal = np.array(y_signal, dtype='float')
    fft_y = np.fft.rfft(y_signal)
    n = int(len(y_signal))
    abs_fy = np.abs(fft_y) / n * 2
    freqs = np.linspace(0, fs / 2, int(n / 2 + 1))
    frqp = freqs[np.argmax(abs_fy)]
    return freqs.tolist(), abs_fy.tolist(), float(frqp)


def wind_spectrum(wind_velocity, h_angle, attack_angle, sample_frq=1):
    '''

    :param wind_velocity:
    :param attack_angle:
    :param sample_frq:
    :return:
    '''
    if type(wind_velocity) is not np.ndarray:
        wind_velocity = np.array(wind_velocity, dtype='float')
    if type(attack_angle) is not np.ndarray:
        h_angle = np.array(h_angle, dtype='float')
    if type(attack_angle) is not np.ndarray:
        attack_angle = np.array(attack_angle, dtype='float')
    h_angle = np.deg2rad(h_angle)
    attack_angle = np.deg2rad(attack_angle)
    wind_h = wind_velocity * np.sin(attack_angle)
    main_angle = sum(wind_h * h_angle) / sum(wind_h)
    wind_h = wind_h * np.cos(h_angle - main_angle)
    par1, par2, wind_pulse, par3 = wind_stats(wind_h, sample_frq=1)
    wind_pulse = np.abs(wind_pulse)
    fft_y = np.fft.rfft(wind_pulse)
    n = int(len(wind_pulse))
    abs_fy = (np.abs(fft_y) * 2) ** 2 / n
    freqs = np.linspace(0, sample_frq / 2, int(n / 2 + 1))
    return freqs.tolist(), abs_fy.tolist()


if __name__ == "__main__":
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float')
    b = movmean(data, 4)
    print(b)
    c = movstd(data, 4)
    print(c)

    wind_velocity = np.random.random(2048)-0.5
    h_angle = np.random.randn(2048)
    h_angle = h_angle / np.max(np.abs(h_angle))
    h_angle = h_angle * 85
    attack_angle = np.random.randn(2048)
    attack_angle = attack_angle / np.max(np.abs(attack_angle))
    attack_angle = attack_angle * 85
    freqs, abs_fy = wind_spectrum(wind_velocity, h_angle, attack_angle, sample_frq=1)
    plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    plt.title('wind')
    # plt.scatter(x_signal, y_signal, color='b', label='train set')
    plt.plot(wind_velocity, color='r', label='wind_spectrum')
    plt.legend(loc='lower right')  # label面板放到figure的右下角
    plt.show()
    plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    plt.title('wind_spectrum')
    # plt.scatter(x_signal, y_signal, color='b', label='train set')
    plt.plot(freqs, abs_fy, color='r', label='wind_spectrum')
    plt.legend(loc='lower right')  # label面板放到figure的右下角
    plt.show()
