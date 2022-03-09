# -*- coding: utf-8 -*-
import numpy as np
import math

main_path = r".\vertex_vibration\\"


def vv_probability(acc_data, frq=50):
    pass




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


def power_spectrum(data, frq):
    pass


def find_peaks(data):
    pass


if __name__ == "__main__":
    from scipy import signal
    import numpy as np

    xs = np.arange(0, np.pi, 0.05)
    data = np.sin(xs)
    peakind = signal.find_peaks_cwt(data, np.arange(1, 10))
    peakind, xs[peakind], data[peakind]