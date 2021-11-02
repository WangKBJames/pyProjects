from dataReader import data_reader, data_location

main_path = r"G:\JSTI\数据分析组文件\江阴位移\wy202004"
sensor_num = "WY070102"
sample_frq = 1
t_start_list = [2020, 4, 13, 10, 0, 0]
t_end_list = [2020, 4, 13, 10, 50, 0]
t_list, data = data_reader(main_path, sensor_num, t_start_list, t_end_list, sample_frq)
print(len(data))
print(len(t_list))
