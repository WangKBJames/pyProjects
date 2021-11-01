# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from binReader import bin_reader
from timeParser import time_parser, time_list
import matplotlib.dates as mdates


a = r"E:\RecycleBin~1cdd.ffs_tmp\data\新建文件夹\南京三桥数据201801\WD\温度\2018\01\22\WD010101_200009.WD"
y = bin_reader(a)
t = time_parser(a)
tl = time_list(t, len(y), 1 / 60)

plt.plot(len1)
plt.show()
