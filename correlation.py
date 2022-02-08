import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

'''

计算相关函数
输入：
x_signal：float[]，时程数据1，多为加速度
y_signal：float[]，时程数据2，多为加速度
sample_frequency： float，数据x_signal、y_signal的采样频率，如加速度为20或50Hz
输出：
x_fit: float[]，频率，单位：Hz
y_fit：float[]，相干函数
a: float，相关函y=a*x+b,系数k
b: float，相关函y=a*x+b,a
corr：float，相关系数

'''


def correlation(x_signal, y_signal):
    if type(x_signal) is not np.ndarray:
        x_signal = np.array(x_signal, dtype='float')
    if type(x_signal) is not np.ndarray:
        y_signal = np.array(y_signal, dtype='float')

    frq, cxy = signal.coherence(x_signal, y_signal, sample_frequency, nperseg=1024)
    return frq.tolist(), cxy.tolist(), np.float(cxy.max()), np.float(cxy.min()), np.float(cxy.mean())


# -*- coding: utf-8 -*-
"""
@author: 蔚蓝的天空TOM
Talk is cheap, show me the code
Aim：最小二乘法库函数leastsq使用示例详解
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


# hypothesis function
def hypothesis_func(w, x):
    w1, w0 = w
    return w1 * x + w0


# error function
def error_func(w, train_x, train_y, msg):
    print(msg)
    return hypothesis_func(w, train_x) - train_y


def dump_fit_func(w_fit):
    w1, w0 = w_fit
    print("fitting line=", str(w1) + "*x + " + str(w0))
    return


# square error平方差函数
def dump_fit_cost(w_fit, train_x, train_y):
    error = error_func(w_fit, train_x, train_y, "")
    square_error = sum(e * e for e in error)
    print('fitting cost:', str(square_error))
    return square_error


# main function
if __name__ == "__main__":
    # train set
    train_x = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
    train_y = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])

    # linear regression by leastsq
    msg = "invoke scipy leastsq"
    w_init = [20, 1]  # weight factor init
    fit_ret = leastsq(error_func, w_init, args=(train_x, train_y, msg))
    w_fit = fit_ret[0]

    # dump fit result
    dump_fit_func(w_fit)
    fit_cost = dump_fit_cost(w_fit, train_x, train_y)

    # test set
    test_x = np.array(np.arange(train_x.min(), train_x.max(), 1.0))
    test_y = hypothesis_func(w_fit, test_x)

    # show result by figure
    plt.figure(1)
    plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    plt.title('linear regression by scipy leastsq')
    plt.scatter(train_x, train_y, color='b', label='train set')
    plt.scatter(test_x, test_y, color='r', marker='^', label='test set', linewidth=2)
    plt.plot(test_x, test_y, color='r', label='fitting line')
    plt.legend(loc='lower right')  # label面板放到figure的右下角

    plt.show()


# if __name__ == "__main__":
#     fs = 10e3
#     N = 1e5
#     amp = 20
#     freq = 1234.0
#     noise_power = 0.001 * fs / 2
#     time = np.arange(N) / fs
#     b, a = signal.butter(2, 0.25, 'low')
#     x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
#     y = signal.lfilter(b, a, x)
#     x += amp * np.sin(2 * np.pi * freq * time)
#     y += np.random.normal(scale=0.1 * np.sqrt(noise_power), size=time.shape)
#     frq, cxy, cxymax, cxymin, cxymean = coherence(x, y, fs)
#     # f, Cxy = signal.coherence(x, y, fs, nperseg=1024)
#     plt.semilogy(f.tolist(), Cxy.tolist())
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('Coherence')
#     plt.show()