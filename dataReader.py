import os
import glob

def data_reader(main_path, sensor_num, t_start, t_end, sample_frq):
    pass


def data_location(main_path, sensor_num, t):
    if len(t) <= 3:
        pass
    year_str = '%04d ' % t[0]
    month_str = '%02d ' % t[1]
    day_str = '%02d ' % t[2]
    data_folder = main_path.join(['\\', year_str, '\\', month_str, '\\', day_str, '\\', sensor_num, '_*.*'])
    file_list = glob.glob(data_folder)
    return file_list
    # file_list = os.listdir()
