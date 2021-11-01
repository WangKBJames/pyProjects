# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.rcParams['font.sans-serif'] = ['SimHei']                           # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False                             # 用来正常显示负号

fig=plt.figure(figsize=(12,6))                                          # 定义图并设置画板尺寸
fig.set(alpha=0.2)                                                      # 设定图表颜色alpha参数
# fig.tight_layout()                                                    # 调整整体空白
plt.subplots_adjust(bottom=0.06,top=0.94,left=0.08,right=0.94,wspace =0.36, hspace =0.5)       # 设置作图范围、子图间距。

df_milano=pd.read_csv("milano_270615.csv")                            # 读取数据

x1= df_milano['day'].values                                                     # 自变量序列
x1= [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in x1]                     # 格式化时间数据输入
y1= df_milano['temp']                                                           # 因变量序列
ax=fig.add_subplot(111)                                                         # 定义子图
plt.xticks(rotation=70)                                                         # 横坐标刻度旋转角度
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))                  # 设置横坐标时间标签的格式
# ax.xaxis.set_major_locator(mdates.HourLocator())                              # 指定横坐标刻度序列
ax.set_xticks(x1)                                                               # 指定横坐标刻度序列
ax.plot(x1,y1,'r')                                                              # 绘图

plt.show()