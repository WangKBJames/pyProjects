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
            y_mean = np.append(y_mean, np.mean(y_signal[i- int(math.ceil(win_n / 2)) + 1:]))
        else:
            y_mean = np.append(y_mean, np.mean(y_signal[i - math.floor(win_n / 2):i + int(math.ceil(win_n / 2))]))
    return y_mean.tolist()


def wind_stats(wind_velocity, horizontal_angle, attack_angle, sample_frq=1):
    '''
    :param wind_velocity:
    :param horizontal_angle:
    :param attack_angle:
    :param sample_frq:
    :return:
    '''
    pass
