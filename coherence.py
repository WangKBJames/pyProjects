import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

'''

计算相干函数
输入：
x_signal：float[]，时程数据1，多为加速度
y_signal：float[]，时程数据2，多为加速度
sample_frequency： float，数据x_signal、y_signal的采样频率，如加速度为20或50Hz
输出：
frq: float[]，频率，单位：Hz
cxy：float[]，相干函数
cxymax: float，相干函数最大值
cxymin：float，相干函数最小值
cxymean：float，相干函数平均值

'''


def coherence(x_signal, y_signal, sample_frequency):
    if type(x_signal) is not np.ndarray:
        x_signal = np.array(x_signal, dtype='float')
    if type(x_signal) is not np.ndarray:
        y_signal = np.array(y_signal, dtype='float')
    frq, cxy = signal.coherence(x_signal, y_signal, sample_frequency, nperseg=1024)
    return frq.tolist(), cxy.tolist(), np.float(cxy.max()), np.float(cxy.min()), np.float(cxy.mean())


if __name__ == "__main__":
    fs = 10e3
    N = 1e5
    amp = 20
    freq = 1234.0
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    b, a = signal.butter(2, 0.25, 'low')
    x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    y = signal.lfilter(b, a, x)
    x += amp * np.sin(2 * np.pi * freq * time)
    y += np.random.normal(scale=0.1 * np.sqrt(noise_power), size=time.shape)
    frq, cxy, cxymax, cxymin, cxymean = coherence(x, y, fs)
    # f, Cxy = signal.coherence(x, y, fs, nperseg=1024)
    plt.semilogy(f.tolist(), Cxy.tolist())
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.show()
