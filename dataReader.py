import datetime
import glob
from binReader import bin_reader, gnss_reader, wind_reader
from timeParser import time_parser, time_list
import calendar
import numpy as np


def bin_data(main_path, sensor_num, t_start_list, t_end_list, sample_frq):
    t_start = datetime.datetime(*t_start_list)
    t_end = datetime.datetime(*t_end_list)
    t_delta = t_end - t_start
    s_delta = t_end.timestamp() - t_start.timestamp()
    t_list = []
    data = []
    if s_delta < 0:
        print('=======> 查询结速时间早于开始时间，查询异常！')
        return t_list, data
    if (t_start.day == t_end.day) & (t_start.month == t_end.month) & (t_start.year == t_end.year):
        if t_start.hour == t_end.hour:
            t = t_start_list[0:4]
            file_list = data_location(main_path, sensor_num, t)
            if len(file_list) > 0:
                data = bin_reader(file_list[0])
            if len(data) > 0:
                t_datetime = time_parser(file_list[0])
                t_list = time_list(t_datetime, len(data), sample_frq)
        else:

            for i in range(t_start.hour, t_end.hour + 1):
                ti = t_start.replace(hour=i)
                t = [ti.year, ti.month, ti.day, ti.hour]
                file_list = data_location(main_path, sensor_num, t)
                if len(file_list) > 0:
                    data_i = bin_reader(file_list[0])
                if len(data_i) > 0:
                    data.extend(data_i)
                    t_datetime = time_parser(file_list[0])
                    t_list_i = time_list(t_datetime, len(data_i), sample_frq)
                    t_list.extend(t_list_i)
    else:
        for i in range(t_start.year, t_end.year + 1):
            if i == t_start.year:
                month_start_num = t_start.month
                if i == t_end.year:
                    month_end_num = t_end.month
                else:
                    month_end_num = int(12)
            elif i == t_end.year:
                month_end_num = t_end.month
                if i == t_start.year:
                    month_start_num = t_start.month
                else:
                    month_start_num = int(1)
            else:
                month_start_num = int(1)
                month_end_num = int(12)
            for j in range(month_start_num, month_end_num + 1):
                if (j == t_start.month) & (i == t_start.year):
                    day_start_num = t_start.day
                    if (j == t_end.month) & (i == t_end.year):
                        day_end_num = t_end.day
                    else:
                        day_end_num = calendar.monthrange(i, j)[1]
                elif (j == t_end.month) & (i == t_end.year):
                    day_end_num = t_end.day
                    if (j == t_start.month) & (i == t_start.year):
                        day_start_num = t_start.day
                    else:
                        day_start_num = int(1)
                else:
                    day_start_num = int(1)
                    day_end_num = calendar.monthrange(i, j)[1]
                for k in range(day_start_num, day_end_num + 1):
                    if (k == t_start.day) & (j == t_start.month) & (i == t_start.year):
                        h_start_num = t_start.hour
                        h_end_num = int(23)
                    elif (k == t_end.day) & (j == t_end.month) & (i == t_end.year):
                        h_start_num = int(0)
                        h_end_num = t_end.hour
                    else:
                        h_start_num = int(0)
                        h_end_num = int(23)
                    if (h_start_num == 0) & (h_end_num == 23):
                        t = [i, j, k]
                        file_list = data_location(main_path, sensor_num, t)
                    else:
                        file_list = []
                        for h in range(h_start_num, h_end_num + 1):
                            t = [i, j, k, h]
                            file_list.extend(data_location(main_path, sensor_num, t))
                    if len(file_list) > 0:
                        for file_str in file_list:
                            data_i = bin_reader(file_str)
                            if len(data_i) > 0:
                                if len(data_i) > sample_frq * 3600:
                                    data_i = data_i[0:sample_frq * 3600]
                                elif len(data_i) < sample_frq * 3600:
                                    data_i.extend(np.full(int(sample_frq * 3600 - len(data_i)), np.nan))
                                    # print(data_i[-1])
                            data.extend(data_i)
                            t_datetime = time_parser(file_str)
                            t_list_i = time_list(t_datetime, len(data_i), sample_frq)
                            t_list.extend(t_list_i)

    return t_list, data


def gnss_data(main_path, sensor_num, t_start_list, t_end_list, return_ref=[0, 1, 2], sample_frq=1):
    t_start = datetime.datetime(*t_start_list)
    t_end = datetime.datetime(*t_end_list)
    s_delta = t_end.timestamp() - t_start.timestamp()
    t_list = []
    data = []
    if s_delta < 0:
        print('=======> 查询结速时间早于开始时间，查询异常！')
        return t_list, data
    if (t_start.day == t_end.day) & (t_start.month == t_end.month) & (t_start.year == t_end.year):
        if t_start.hour == t_end.hour:
            t = t_start_list[0:4]
            file_list = data_location(main_path, sensor_num, t)
            if len(file_list) > 0:
                t_list_i, *data_i = gnss_reader(file_list[0], return_ref)
                t_list.extend(t_list_i)
                data = data_i
        else:
            for i in range(t_start.hour, t_end.hour + 1):
                ti = t_start.replace(hour=i)
                t = [ti.year, ti.month, ti.day, ti.hour]
                file_list = data_location(main_path, sensor_num, t)
                if len(file_list) > 0:
                    t_list_i, *data_i = gnss_reader(file_list[0], return_ref)
                    t_list.extend(t_list_i)
                    if not data:
                        for j in range(len(data)):
                            data[j].extend(data_i[j])
                    else:
                        data = data_i
    else:
        for i in range(t_start.year, t_end.year + 1):
            if i == t_start.year:
                month_start_num = t_start.month
                if i == t_end.year:
                    month_end_num = t_end.month
                else:
                    month_end_num = int(12)
            elif i == t_end.year:
                month_end_num = t_end.month
                if i == t_start.year:
                    month_start_num = t_start.month
                else:
                    month_start_num = int(1)
            else:
                month_start_num = int(1)
                month_end_num = int(12)
            for j in range(month_start_num, month_end_num + 1):
                if (j == t_start.month) & (i == t_start.year):
                    day_start_num = t_start.day
                    if (j == t_end.month) & (i == t_end.year):
                        day_end_num = t_end.day
                    else:
                        day_end_num = calendar.monthrange(i, j)[1]
                elif (j == t_end.month) & (i == t_end.year):
                    day_end_num = t_end.day
                    if (j == t_start.month) & (i == t_start.year):
                        day_start_num = t_start.day
                    else:
                        day_start_num = int(1)
                else:
                    day_start_num = int(1)
                    day_end_num = calendar.monthrange(i, j)[1]
                for k in range(day_start_num, day_end_num + 1):
                    if (k == t_start.day) & (j == t_start.month) & (i == t_start.year):
                        h_start_num = t_start.hour
                        h_end_num = int(23)
                    elif (k == t_end.day) & (j == t_end.month) & (i == t_end.year):
                        h_start_num = int(0)
                        h_end_num = t_end.hour
                    else:
                        h_start_num = int(0)
                        h_end_num = int(23)
                    if (h_start_num == 0) & (h_end_num == 23):
                        t = [i, j, k]
                        file_list = data_location(main_path, sensor_num, t)
                    else:
                        file_list = []
                        for h in range(h_start_num, h_end_num + 1):
                            t = [i, j, k, h]
                            file_list.extend(data_location(main_path, sensor_num, t))
                    if len(file_list) > 0:
                        for file_str in file_list:
                            t_list_i, *data_i = gnss_reader(file_str, return_ref)
                            t_list.extend(t_list_i)
                            if len(data_i[0]) > 0:
                                if len(data_i[0]) > sample_frq * 3600:
                                    for ind in range(len(data_i)):
                                        data_i[ind] = data_i[ind][0:sample_frq * 3600]
                                elif len(data_i[0]) < sample_frq * 3600:
                                    for ind in range(len(data_i)):
                                        data_i[ind].extend(np.full(int(sample_frq * 3600 - len(data_i[ind])), np.nan))
                                    # print(data_i[-1])
                            if data:
                                for ind in range(len(data)):
                                    data[ind].extend(data_i[ind])
                            else:
                                data = data_i

    return t_list, data


def wind_data(main_path, sensor_num, t_start_list, t_end_list, sample_frq=1):
    t_start = datetime.datetime(*t_start_list)
    t_end = datetime.datetime(*t_end_list)
    t_delta = t_end - t_start
    s_delta = t_end.timestamp() - t_start.timestamp()
    t_list = []
    fsh, fsk, alpha, beta = [], [], [], []
    if s_delta < 0:
        print('=======> 查询结速时间早于开始时间，查询异常！')
        return t_list, fsh, fsk, alpha, beta
    if (t_start.day == t_end.day) & (t_start.month == t_end.month) & (t_start.year == t_end.year):
        if t_start.hour == t_end.hour:
            t = t_start_list[0:4]
            file_list = data_location(main_path, sensor_num, t)
            if len(file_list) > 0:
                fsh, fsk, alpha, beta = wind_reader(file_list[0], return_ref=[0, 1, 2, 3])
            if len(fsh) > 0:
                t_datetime = time_parser(file_list[0])
                t_list = time_list(t_datetime, len(fsh), sample_frq)
        else:
            for i in range(t_start.hour, t_end.hour + 1):
                ti = t_start.replace(hour=i)
                t = [ti.year, ti.month, ti.day, ti.hour]
                file_list = data_location(main_path, sensor_num, t)
                if len(file_list) > 0:
                    fsh_i, fsk_i, alpha_i, beta_i = wind_reader(file_list[0], return_ref=[0, 1, 2, 3])
                if len(fsh_i) > 0:
                    fsh.extend(fsh_i)
                    fsk.extend(fsk_i)
                    alpha.extend(alpha_i)
                    beta.extend(beta_i)
                    t_datetime = time_parser(file_list[0])
                    t_list_i = time_list(t_datetime, len(fsh_i), sample_frq)
                    t_list.extend(t_list_i)
    else:
        for i in range(t_start.year, t_end.year + 1):
            if i == t_start.year:
                month_start_num = t_start.month
                if i == t_end.year:
                    month_end_num = t_end.month
                else:
                    month_end_num = int(12)
            elif i == t_end.year:
                month_end_num = t_end.month
                if i == t_start.year:
                    month_start_num = t_start.month
                else:
                    month_start_num = int(1)
            else:
                month_start_num = int(1)
                month_end_num = int(12)
            for j in range(month_start_num, month_end_num + 1):
                if (j == t_start.month) & (i == t_start.year):
                    day_start_num = t_start.day
                    if (j == t_end.month) & (i == t_end.year):
                        day_end_num = t_end.day
                    else:
                        day_end_num = calendar.monthrange(i, j)[1]
                elif (j == t_end.month) & (i == t_end.year):
                    day_end_num = t_end.day
                    if (j == t_start.month) & (i == t_start.year):
                        day_start_num = t_start.day
                    else:
                        day_start_num = int(1)
                else:
                    day_start_num = int(1)
                    day_end_num = calendar.monthrange(i, j)[1]
                for k in range(day_start_num, day_end_num + 1):
                    if (k == t_start.day) & (j == t_start.month) & (i == t_start.year):
                        h_start_num = t_start.hour
                        h_end_num = int(23)
                    elif (k == t_end.day) & (j == t_end.month) & (i == t_end.year):
                        h_start_num = int(0)
                        h_end_num = t_end.hour
                    else:
                        h_start_num = int(0)
                        h_end_num = int(23)
                    if (h_start_num == 0) & (h_end_num == 23):
                        t = [i, j, k]
                        file_list = data_location(main_path, sensor_num, t)
                    else:
                        file_list = []
                        for h in range(h_start_num, h_end_num + 1):
                            t = [i, j, k, h]
                            file_list.extend(data_location(main_path, sensor_num, t))
                    if len(file_list) > 0:
                        for file_str in file_list:
                            fsh_i, fsk_i, alpha_i, beta_i = wind_reader(file_str, return_ref=[0, 1, 2, 3])
                            if len(fsh_i) > 0:
                                if len(fsh_i) > sample_frq * 3600:
                                    fsh_i = fsh_i[0:sample_frq * 3600]
                                    fsk_i = fsk_i[0:sample_frq * 3600]
                                    alpha_i = alpha_i[0:sample_frq * 3600]
                                    beta_i = beta_i[0:sample_frq * 3600]
                                elif len(fsh_i) < sample_frq * 3600:
                                    fsh_i.extend(np.full(int(sample_frq * 3600 - len(fsh_i)), np.nan))
                                    fsk_i.extend(np.full(int(sample_frq * 3600 - len(fsk_i)), np.nan))
                                    alpha_i.extend(np.full(int(sample_frq * 3600 - len(alpha_i)), np.nan))
                                    beta_i.extend(np.full(int(sample_frq * 3600 - len(beta_i)), np.nan))
                                    # print(data_i[-1])
                            fsh.extend(fsh_i)
                            fsk.extend(fsk_i)
                            alpha.extend(alpha_i)
                            beta.extend(beta_i)
                            t_datetime = time_parser(file_str)
                            t_list_i = time_list(t_datetime, len(fsh_i), sample_frq)
                            t_list.extend(t_list_i)

    return t_list, fsh, fsk, alpha, beta


def data_location(main_path, sensor_num, t):
    year_str = '%04d' % t[0]
    month_str = '%02d' % t[1]
    day_str = '%02d' % t[2]
    if len(t) > 3:
        hh_str = '%02d' % t[3]
    else:
        hh_str = '*'
    str_list = [main_path, '\\', year_str, '\\', month_str, '\\', day_str,
                '\\', sensor_num, '_', hh_str, '*.*']
    data_folder = ""
    for i in str_list:
        data_folder = data_folder + i
    file_list = glob.glob(data_folder)
    return file_list


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    main_path = r"I:\JSTI\数据\江阴\江阴数据\FS"
    sensor_num = "FS060101"
    sample_frq = 1
    t_start_list = [2021, 12, 25, 17, 0, 0]
    t_end_list = [2021, 12, 26, 5, 10, 0]
    t_list, fsh, fsk, alpha, beta = wind_data(main_path, sensor_num, t_start_list, t_end_list, sample_frq=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure(figsize=(12, 6))  # 定义图并设置画板尺寸
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数
    # fig.tight_layout()                                                    # 调整整体空白
    plt.subplots_adjust(bottom=0.25, top=0.94, left=0.08, right=0.94, wspace=0.36, hspace=0.5)
    ax = fig.add_subplot(111)  # 定义子图
    plt.xticks(rotation=90)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
    ax.plot(t_list, fsh, 'b')
    plt.show()
