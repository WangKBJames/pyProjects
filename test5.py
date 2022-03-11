from dataReader import bin_data, data_location, gnss_data
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# main_path = r"G:\JSTI\数据分析组文件\江阴位移\wy202004"
# sensor_num = "WY070102"
# sample_frq = 1
# t_start_list = [2020, 4, 7, 3, 0, 0]
# t_end_list = [2020, 4, 12, 23, 0, 0]
# main_path = r"G:\JSTI\数据分析组文件\徐州和平桥\和平桥数据\data\LW\路面温度"
# sensor_num = "LW010101"
# sample_frq = 1
# t_start_list = [2019, 9, 20, 12, 25, 0]
# t_end_list = [2019, 10, 23, 13, 10, 0]
# t1 = time.time()
# t_list, data = bin_data(main_path, sensor_num, t_start_list, t_end_list, sample_frq)
# t2 = time.time()
# dt = t2 - t1
# print(len(data) / 3600 / 24)
# print(len(t_list))
# print(t_list[0])
# print(t_list[-1])
# print("运行时间为%.2fs" % dt)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# fig = plt.figure(figsize=(12, 6))  # 定义图并设置画板尺寸
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
# # fig.tight_layout()                                                    # 调整整体空白
# plt.subplots_adjust(bottom=0.25, top=0.94, left=0.08, right=0.94, wspace=0.36, hspace=0.5)
# ax = fig.add_subplot(111)  # 定义子图
# plt.xticks(rotation=90)
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
# ax.plot(t_list, data, 'r')
# plt.show()

main_path = r"Z:\江苏控股桥群\江阴\gps"
sensor_num = "BD030102"
sample_frq = 1
t_start_list = [2021, 8, 1, 0, 0, 0]
t_end_list = [2021, 8, 31, 23, 0, 0]
t1 = time.time()
t_list, data = gnss_data(main_path, sensor_num, t_start_list, t_end_list)
t2 = time.time()
# print(len(t_list))
# print(len(data[0]))
dt = t2 - t1
print(len(data[2]))
print(len(t_list))
# print(t_list[0])
# print(t_list[-1])
print("运行时间为%.2fs" % dt)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
fig = plt.figure(figsize=(12, 6))  # 定义图并设置画板尺寸
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
# fig.tight_layout()                                                    # 调整整体空白
plt.subplots_adjust(bottom=0.25, top=0.94, left=0.08, right=0.94, wspace=0.36, hspace=0.5)
ax = fig.add_subplot(111)  # 定义子图
plt.xticks(rotation=90)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
ax.plot(data[2], 'r')
plt.show()
