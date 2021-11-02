# -*- coding: utf-8 -*-
import time
import datetime
import re


def time_list(start_t, len_t, sample_frq):
    tl = [start_t+datetime.timedelta(seconds=i*1/sample_frq) for i in range(0, len_t)]
    return tl


def time_parser(file_path):
    # time_str: 时间字符串，例如：
    # “I:\JSTI\数据分析组文件\江阴2021中铁桥隧升级改造样本\数据\数据\原始数据\data\WD\2021\09\01\WD060201_210000.WD”
    # a = r"I:\JSTI\数据分析组文件\江阴2021中铁桥隧升级改造样本\数据\数据\原始数据\data\WD\2021\09\01\WD060201_210000.WD"
    match_obj = re.match(r'.*\\(\d{4})\\(\d{2})\\(\d{2})\\.*_(\d{6})\..*', file_path, re.M | re.I)
    year = match_obj.group(1)
    month = match_obj.group(2)
    day = match_obj.group(3)
    HH = match_obj.group(4)[0:2]
    MM = match_obj.group(4)[2:4]
    SS = match_obj.group(4)[4:]
    dt = year+"-"+month+"-"+day+" "+HH+":"+MM+":"+SS
    return datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    # print(datetime.strptime(dt, '%Y-%m-%d %H:%M:%S'))



