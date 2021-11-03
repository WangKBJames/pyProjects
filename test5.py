from dataReader import data_reader, data_location
import time

# main_path = r"G:\JSTI\数据分析组文件\江阴位移\wy202004"
# sensor_num = "WY070102"
# sample_frq = 1
# t_start_list = [2020, 4, 7, 3, 0, 0]
# t_end_list = [2020, 4, 12, 23, 0, 0]
main_path = r"I:\JSTI\数据分析组文件\徐州和平桥\和平桥数据\data\LW\路面温度"
sensor_num = "LW010101"
sample_frq = 1
t_start_list = [2019, 9, 20, 12, 0, 0]
t_end_list = [2020, 2, 20, 11, 0, 0]
t1 = time.time()
t_list, data = data_reader(main_path, sensor_num, t_start_list, t_end_list, sample_frq)
t2 = time.time()
dt = t2 - t1
print(len(data))
print(len(t_list))
print(t_list[-1])
print("运行时间为%.2fs" % dt)
