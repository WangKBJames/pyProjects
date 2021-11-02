import datetime
import glob
from binReader import bin_reader
from timeParser import time_parser, time_list


def data_reader(main_path, sensor_num, t_start_list, t_end_list, sample_frq):
    t_start = datetime.datetime(*t_start_list)
    t_end = datetime.datetime(*t_end_list)
    t_delta = t_end - t_start
    s_delta = t_delta.seconds
    if s_delta < 0:
        print('=======> 查询结速时间早于开始时间，查询异常！')
        return [], []
    if t_start.day == t_end.day:
        if t_start.hour == t_end.hour:
            t = t_start_list[0:4]
            file_list = data_location(main_path, sensor_num, t)
            data = bin_reader(file_list[0])
            if len(data) > 0:
                t_datetime = time_parser(file_list[0])
                t_list = time_list(t_datetime, len(data), sample_frq)
            else:
                t_list = []
        else:
            data = []
            t_list = []
            for i in range(t_start.hour, t_end.hour + 1):
                ti = t_start.replace(hour=i)
                t = [ti.year, ti.month, ti.day, ti.hour]
                file_list = data_location(main_path, sensor_num, t)
                data_i = bin_reader(file_list[0])
                if len(data_i) > 0:
                    data.extend(data_i)
                    t_datetime = time_parser(file_list[0])
                    # print(t_datetime)
                    t_list_i = time_list(t_datetime, len(data_i), sample_frq)
                    # print(len(data))
                    t_list.extend(t_list_i)
    else:
        pass

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
