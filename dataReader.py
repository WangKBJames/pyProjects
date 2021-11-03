import datetime
import glob
from binReader import bin_reader
from timeParser import time_parser, time_list
import calendar


def data_reader(main_path, sensor_num, t_start_list, t_end_list, sample_frq):
    t_start = datetime.datetime(*t_start_list)
    t_end = datetime.datetime(*t_end_list)
    t_delta = t_end - t_start
    s_delta = t_delta.seconds
    data = []
    t_list = []
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
                                    data.extend(data_i)
                                    t_datetime = time_parser(file_list[0])
                                    t_list_i = time_list(t_datetime, len(data_i), sample_frq)
                                    t_list.extend(t_list_i)

    return t_list, data


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
