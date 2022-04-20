import numpy as np
from scipy import signal

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

main_path = r".\coherence\\"


def coherence(x_signal, y_signal, sample_frequency):
    '''

    :param x_signal:
    :param y_signal:
    :param sample_frequency:
    :return:
    '''
    if type(x_signal) is not np.ndarray:
        x_signal = np.array(x_signal, dtype='float')
    x_signal[np.isnan(x_signal)] = np.nanmean(x_signal)
    if type(x_signal) is not np.ndarray:
        y_signal = np.array(y_signal, dtype='float')
    y_signal[np.isnan(y_signal)] = np.nanmean(y_signal)
    if len(x_signal) > len(y_signal):
        x_signal = x_signal[:len(y_signal)]
    if len(x_signal) < len(y_signal):
        y_signal = y_signal[:len(x_signal)]
    if len(x_signal) < 2048:
        frq = np.arange(0, sample_frequency / 2048 * 1025, sample_frequency / 2048)
        cxy = np.ones_like(frq) - 1
    else:
        frq, cxy = signal.coherence(x_signal, y_signal, sample_frequency, nperseg=2048)
    return [frq.tolist(), cxy.tolist(), float(cxy.max()), float(cxy.min()), float(cxy.mean())]


def process():
    fs = np.loadtxt(main_path + r"input\par.txt", dtype='float')
    data_1 = np.loadtxt(main_path + r"input\shuju1.txt", dtype='float')
    data_2 = np.loadtxt(main_path + r"input\shuju2.txt", dtype='float')
    frq, cxy, cxy_max, cxy_min, cxy_mean = coherence(data_1, data_2, fs)
    np.savetxt(main_path + r"output\fig2_x.txt", frq)
    np.savetxt(main_path + r"output\fig2_y_1.txt", cxy)
    doc_str = "最大值：{0:.3f}\n" \
              "最小值：{1:.3f}\n" \
              "平均值：{2:.3f}".format(cxy_max, cxy_min, cxy_mean)
    with open(main_path + r"output\shuoming.txt", "w", encoding="utf-8") as f:
        f.write(doc_str)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if False:
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
        plt.semilogy(frq, cxy)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Coherence')
        plt.show()

        np.savetxt(main_path + r"input\shuju1.txt", x)
        np.savetxt(main_path + r"input\shuju2.txt", y)
    if True:
        from dataReader import bin_data

        zd_path = r"W:\ZD\\"
        zd1_sensor = "ZD020116"
        zd2_sensor = "ZD020117"
        t_start_list = [2022, 2, 11, 0, 0, 0]
        t_end_list = [2022, 2, 11, 2, 0, 0]
        _, zd1_data = bin_data(zd_path, zd1_sensor, t_start_list, t_end_list, sample_frq=20)
        _, zd2_data = bin_data(zd_path, zd2_sensor, t_start_list, t_end_list, sample_frq=20)
        np.savetxt(main_path + r"input\shuju1.txt", zd1_data)
        np.savetxt(main_path + r"input\shuju2.txt", zd2_data)
        # coherence(x_signal, y_signal, sample_frequency)
        process()
        data_1 = np.loadtxt(main_path + r"output\fig2_x.txt", dtype='float')
        print(len(data_1))
        data_2 = np.loadtxt(main_path + r"output\fig2_y_1.txt", dtype='float')
        plt.plot(data_1, data_2)
        plt.show()
