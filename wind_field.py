# -*- coding: utf-8 -*-

import numpy as np
import math

# import pandas as pd
# from scipy.fftpack import fft
# from scipy.optimize import leastsq
# import matplotlib.pyplot as plt
# from scipy.optimize import leastsq
# from scipy.stats import pearsonr

main_path = r".\wind_field\\"


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
    wmean_2min = movmean(wind_velocity, int(120 * sample_frq))
    wmean_10min = movmean(wind_velocity, int(600 * sample_frq))
    wpulse = [wind_velocity[i] - wmean_10min[i] for i in range(len(wind_velocity))]
    wstd_10min = movstd(wind_velocity, int(600 * sample_frq))
    return wmean_2min, wmean_10min, wpulse, wstd_10min


def wind_par(wind_velocity, sample_frq=1):
    '''
    计算风场参数，图1 y轴
    :param wind_velocity: float[] 实时风速, 图1，y轴第1条数据（x轴为时间）
    :param sample_frq: float 采样频率，缺省值为1，
    :return: list[]
    list[0]: wmean_2min: float[] 2分钟平均风速时程，单位：m/s,图1，y轴第2条数据（x轴为时间）
    list[1]: wmean_10min:  float[] 10分钟平均风速时程，单位：m/s,图2，y轴第3条数据（x轴为时间）
    list[2]:  wpulse:  float[] 脉动风速时程，单位：m/s,图1，y轴第4条数据（x轴为时间）
    list[3]:  wstd_10min:  float[] 10分钟风速均方根，单位：m/s,图1，y轴第5条数据（x轴为时间）
    list[4]: w_max: float  瞬时风速极值，单位：m/s, 文本区域
    list[5]: w2_max: float  2分钟风速极值，单位：m/s, 文本区域
    list[6]: w10_max: float  10分钟风速极值，单位：m/s, 文本区域
    list[7]: wstd_max: float  10分钟均方根极值，单位：m/s, 文本区域
    list[8]: wm: float  瞬时风速均值，单位：m/s, 文本区域

    '''
    wmean_2min, wmean_10min, wpulse, wstd_10min = wind_stats(wind_velocity, sample_frq=1)

    return [wmean_2min, wmean_10min, wpulse, wstd_10min, float(np.max(wind_velocity)), float(np.max(wmean_2min)),
            float(np.max(wmean_10min)), float(np.max(wstd_10min)), float(np.nanmean(wind_velocity))]


def attack_angle(angle):
    '''
    风攻角计算，图2
    :param angle: float[]  风竖向角数据（风数据最后一列）
    :return:
    attack_angel: float[]  风攻角数据（图3纵坐标，横坐标为时间）
    '''
    # if type(angle) is not np.ndarray:
    #     angle = np.array(angle, dtype='float')
    # angle[np.isnan(angle)] = np.nanmean(angle)
    # angle_attack = angle
    # for i in range(len(angle)):
    #     if angle[i] > 0:
    #         angle_attack[i] = 90 - angle[i]
    #     else:
    #         angle_attack[i] = -90 - angle[i]
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
    风谱计算 图4
    :param wind_velocity: float[] 实时风速数据（风速数据第2列）
    :param attack_angle: float[] 风水平角数据（风速数据第3列）
    :param sample_frq: float[] 数向角数据（风速数据第4列）
    :return:
    frq：float[] 频率 图3的横坐标，单位(Hz)
    ffy：float[] 频谱数据 图3的纵坐标，单位：(幅值)
    '''
    if type(wind_velocity) is not np.ndarray:
        wind_velocity = np.array(wind_velocity, dtype='float')
    wind_velocity[np.isnan(wind_velocity)] = np.nanmean(wind_velocity)
    if type(attack_angle) is not np.ndarray:
        h_angle = np.array(h_angle, dtype='float')
    h_angle[np.isnan(h_angle)] = np.nanmean(h_angle)
    h_angle = np.deg2rad(h_angle)
    if type(attack_angle) is not np.ndarray:
        attack_angle = np.array(attack_angle, dtype='float')
    attack_angle[np.isnan(attack_angle)] = np.nanmean(attack_angle)
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
    return [freqs[1:].tolist(), abs_fy[1:].tolist()]


def turbulence(wind_velocity, h_angle, attack_angle, sample_frq=1):
    '''
    紊流强度、阵风因子计算，图5，图6
    :param wind_velocity: float[] 实时风速数据（风速数据第2列）
    :param h_angle:  float[] 风水平角数据（风速数据第3列）
    :param attack_angle:  float[] 数向角数据（风速数据第4列）
    :param sample_frq: float 采样频率，缺省值为1
    :return: list
    list[0]: tur_intensity: float[] 湍流强度（无单位），图4纵坐标，（横坐标为时间）
    list[1]: gustiness_factor: float[] 阵风系数（无单位），图5纵坐标，（横坐标为时间）
    '''
    if type(wind_velocity) is not np.ndarray:
        wind_velocity = np.array(wind_velocity, dtype='float')
    wind_velocity[np.isnan(wind_velocity)] = np.nanmean(wind_velocity)
    if type(attack_angle) is not np.ndarray:
        h_angle = np.array(h_angle, dtype='float')
    h_angle[np.isnan(h_angle)] = np.nanmean(h_angle)
    h_angle = np.deg2rad(h_angle)
    if type(attack_angle) is not np.ndarray:
        attack_angle = np.array(attack_angle, dtype='float')
    attack_angle[np.isnan(attack_angle)] = np.nanmean(attack_angle)
    attack_angle = np.deg2rad(attack_angle)
    wind_h = wind_velocity * np.sin(attack_angle)
    main_angle = sum(wind_h * h_angle) / sum(wind_h)
    wind_h = np.abs(wind_h * np.cos(h_angle - main_angle))
    fs2, fs10, wind_pulse, f_std = wind_stats(wind_h, sample_frq=1)
    fs3s = movmean(wind_h, 3 * sample_frq)
    tur_intensity = [f_std[i] / fs10[i] for i in range(len(f_std))]  # 紊流强度
    gustiness_factor = [fs3s[i] / fs10[i] for i in range(len(fs3s))]  # 阵风系数
    return [tur_intensity, gustiness_factor]


def wind_scale(fs):
    '''
    计算风力等级
    :param fs:
    :return:
    '''
    if 0.0 <= fs <= 0.25:
        N = 0
    elif 0.25 < fs <= 1.55:
        N = 1
    elif 1.55 < fs <= 3.35:
        N = 2
    elif 3.35 < fs <= 5.45:
        N = 3
    elif 5.45 < fs <= 7.95:
        N = 4
    elif 7.95 < fs <= 10.75:
        N = 5
    elif 10.75 < fs <= 13.85:
        N = 6
    elif 13.85 < fs <= 17.15:
        N = 7
    elif 17.15 < fs <= 20.75:
        N = 8
    elif 20.75 < fs <= 24.45:
        N = 9
    elif 24.45 < fs <= 28.45:
        N = 10
    elif 28.45 < fs <= 32.65:
        N = 11
    elif 32.65 < fs <= 36.95:
        N = 12  # 强台风  一级飓风
    elif 36.95 < fs <= 41.45:
        N = 13  # 强台风  一级飓风
    elif 41.45 < fs <= 46.15:
        N = 14  # 强台风  二级飓风
    elif 46.15 < fs <= 50.95:
        N = 15  # 强台风  三级飓风
    elif 50.95 < fs <= 56.05:
        N = 16  # 超强台风  三级飓风
    elif 56.05 < fs <= 61.25:
        N = 17  # 超强台风  四级飓风
    elif 61.25 < fs <= 69.4:
        N = 18  # 超强台风 四级飓风
    elif fs > 69.4:
        N = 19  # 超级台风 五级飓风
    return N


def process():
    wind_velocity = np.loadtxt(main_path + r"input\fengsu.txt", dtype='float')
    h_angle = np.loadtxt(main_path + r"input\shuipingjiao.txt", dtype='float')
    attack_angle = np.loadtxt(main_path + r"input\shuzhijiao.txt", dtype='float')
    wmean_2min, wmean_10min, wpulse, wstd_10min, wind_max, wmean_2min_max, wmean_10min_max, wstd_10min_max, wmean = wind_par(
        wind_velocity, sample_frq=1)
    # attack_angle = attack_angle(attack_angle)
    freqs, abs_fy = wind_spectrum(wind_velocity, h_angle, attack_angle, sample_frq=1)
    tur_intensity, gustiness_factor = turbulence(wind_velocity, h_angle, attack_angle, sample_frq=1)
    w_scale = wind_scale(wmean_2min_max)
    np.savetxt(main_path + r"output\fig1_y_1.txt", wind_velocity)
    np.savetxt(main_path + r"output\fig1_y_2.txt", wmean_2min)
    np.savetxt(main_path + r"output\fig1_y_3.txt", wmean_10min)
    np.savetxt(main_path + r"output\fig1_y_4.txt", wpulse)
    np.savetxt(main_path + r"output\fig1_y_5.txt", wstd_10min)
    np.savetxt(main_path + r"output\fig2_y_1.txt", attack_angle)
    np.savetxt(main_path + r"output\fig3_x.txt", freqs)
    np.savetxt(main_path + r"output\fig3_y_1.txt", abs_fy)
    np.savetxt(main_path + r"output\fig4_y_1.txt", tur_intensity)
    np.savetxt(main_path + r"output\fig5_y_1.txt", gustiness_factor)
    doc_str = "瞬时风速极值：{0:.2f}m/s\n" \
              "2分钟风速极值（风力）：{1:.2f}m/s({2:d}级风)\n" \
              "10分钟风速极值：{3:.2f}m/s\n" \
              "10分钟均方根极值：{4:.2f}m/s\n" \
              "风速均值：{5:.2f}m/s".format(wind_max, wmean_2min_max, w_scale, wmean_10min_max, wstd_10min_max, wmean)
    with open(main_path + r"output\shuoming.txt", "w", encoding="utf-8") as f:
        f.write(doc_str)
    return


if __name__ == "__main__":
    # from dataReader import wind_data
    # import matplotlib.dates as mdates

    # main_path = r"I:\JSTI\数据\江阴\江阴数据\FS"
    # sensor_num = "FS060101"
    # sample_frq = 1
    # t_start_list = [2021, 12, 25, 17, 0, 0]
    # t_end_list = [2021, 12, 26, 5, 10, 0]
    # t_list, fsh, fsk, alpha, beta = wind_data(main_path, sensor_num, t_start_list, t_end_list, sample_frq=1)
    # fs2, fs10, fsp, fsstd, fsmax, fs2max, fs10max, fspmax, fsstdmax = wind_par(fsk, sample_frq=1)
    #
    # freqs, abs_fy = wind_spectrum(fsk, alpha, beta, sample_frq=1)
    # wl, zf = turbulence(fsk, alpha, beta, sample_frq=1)
    # fig = plt.figure(figsize=(30, 12))
    # fig.tight_layout()
    # ax1 = fig.add_subplot(221)
    # plt.xticks(rotation=45)
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
    # ax1.plot(t_list, fsk, color='b', label='real-time')
    # ax1.plot(t_list, fs2, color='r', label='2 minuts mean')
    # ax1.plot(t_list, fs10, color='k', label='10 minuts mean')
    # ax1.plot(t_list, fsp, color='m', label='pulse velocity')
    # ax1.plot(t_list, fsstd, color='k', label='standard deviation')
    # plt.xlabel("time")
    # plt.ylabel("wind velocity(m/s)")
    # plt.legend(loc='lower right')
    # ax2 = fig.add_subplot(222)
    # ax2.plot(freqs, abs_fy, color='b', label='wind_spectrum')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('frequency(Hz)')
    # plt.ylabel('amplitude')
    # plt.title('wind')
    # plt.legend(loc='lower right')
    # ax3 = fig.add_subplot(223)
    # plt.xticks(rotation=45)
    # ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
    # ax3.plot(t_list, wl, color='b', label='real-time')
    # plt.xlabel("time")
    # plt.ylabel("turbulence intensity")
    # ax4 = fig.add_subplot(224)
    # plt.xticks(rotation=45)
    # ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
    # ax4.plot(t_list, zf, color='b', label='real-time')
    # plt.xlabel("time")
    # plt.ylabel("gustiness factor")

    # plt.scatter(x_signal, y_signal, color='b', label='train set')
    # plt.plot(fsk, color='r', label='wind_spectrum')
    # plt.legend(loc='lower right')  # label面板放到figure的右下角
    # plt.show()
    # plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    # plt.title('wind_spectrum')
    # # plt.scatter(x_signal, y_signal, color='b', label='train set')
    # plt.plot(freqs, abs_fy, color='b', label='wind_spectrum')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.legend(loc='lower right')  # label面板放到figure的右下角
    # plt.show()

    # data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float')
    # b = movmean(data, 4)
    # print(b)
    # c = movstd(data, 4)
    # print(c)
    #
    # wind_velocity = np.random.random(2048)-0.5
    # h_angle = np.random.randn(2048)
    # h_angle = h_angle / np.max(np.abs(h_angle))
    # h_angle = h_angle * 85
    # attack_angle = np.random.randn(2048)
    # attack_angle = attack_angle / np.max(np.abs(attack_angle))
    # attack_angle = attack_angle * 85
    # freqs, abs_fy = wind_spectrum(wind_velocity, h_angle, attack_angle, sample_frq=1)
    # plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    # plt.title('wind')
    # # plt.scatter(x_signal, y_signal, color='b', label='train set')
    # plt.plot(wind_velocity, color='r', label='wind_spectrum')
    # plt.legend(loc='lower right')  # label面板放到figure的右下角
    # plt.show()
    # plt.figure(figsize=(8, 6))  # 指定图像比例： 8：6
    # plt.title('wind_spectrum')
    # # plt.scatter(x_signal, y_signal, color='b', label='train set')
    # plt.plot(freqs, abs_fy, color='r', label='wind_spectrum')
    # plt.legend(loc='lower right')  # label面板放到figure的右下角
    # plt.show()

    # wind_velocity = 12 * np.abs(np.random.randn(50000))
    # h_angle = 360 * np.random.random(50000)
    # attack_angle = 10 * (np.random.random(50000) - 0.5)
    # np.savetxt(r".\wind_field\input\fengsu.txt", wind_velocity)
    # np.savetxt(r".\wind_field\input\shuipingjiao.txt", h_angle)
    # np.savetxt(r".\wind_field\input\shuzhijiao.txt", attack_angle)
    process()
