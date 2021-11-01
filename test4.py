# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from binReader import bin_reader
from timeParser import time_parser, time_list
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

a = r"G:\JSTI\数据分析组文件\江阴位移\wy202004\2020\04\13\WY070101_180000.WY"
y = bin_reader(a)
t = time_parser(a)
tl = time_list(t, len(y), 1)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
fig = plt.figure(figsize=(12, 6))  # 定义图并设置画板尺寸
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
# fig.tight_layout()                                                    # 调整整体空白
plt.subplots_adjust(bottom=0.15, top=0.94, left=0.08, right=0.94, wspace=0.36, hspace=0.5)
ax = fig.add_subplot(111)  # 定义子图
plt.xticks(rotation=90)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
ax.xaxis.set_major_formatter
ax.plot(tl, y, 'r')
plt.show()
a=np.arange(10).astype(dtype='float')
a = np.random.randn(3, 3)
a.T