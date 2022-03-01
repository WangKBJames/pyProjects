# -*- coding: utf-8 -*-
import struct
import datetime


def bin_reader(file_path):
    try:
        f = open(file_path, 'rb')
        count = 0  # 已读取字节数
        # 获取文件字符数
        count_bytes = f.seek(0, 2)
        # print(count_bytes)
        f.seek(0)
        # 文件版本号字节长度
        len_ver_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
        ver_str = str(f.read(len_ver_str), encoding="utf-8")
        # print(ver_str)
        # 传感器类型字节长度
        len_type_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
        type_str = str(f.read(len_type_str), encoding="utf-8")
        # print(type_str)
        # 传感器安装位置字节长度
        len_pos_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
        pos_str = str(f.read(len_pos_str))
        # print(pos_str)
        # 文件存储开始时间
        id_start_time = int.from_bytes(f.read(4), byteorder='little', signed=False)
        # print(id_start_time)
        # 传感器编号字节长度
        len_sensor_num = int.from_bytes(f.read(1), byteorder='little', signed=False)
        # print(len_sensor_num)
        # 传感器编号
        sensor_num = str(f.read(len_sensor_num), encoding="utf-8")
        # print(sensor_num)
        # 采样频率
        sample_frq = struct.unpack('f', f.read(4))[0]
        # print(sample_frq)
        # 采样精度
        sample_precision = struct.unpack('f', f.read(4))[0]
        # print(sample_precision)
        # 放大倍数
        scalor = int.from_bytes(f.read(4), byteorder='little', signed=False)
        # print(scalor)
        # 灵敏度
        sensitivity = struct.unpack('f', f.read(4))[0]
        # print(sensitivity)
        # 字符'$', 表示文件头的终结
        f.read(1)
        # print(str(f.read(1), encoding="utf-8"))
        data = list()
        # 数据数量
        count_data = int((count_bytes - f.seek(0, 1)) / 4)
        # print(count_data)
        data = [struct.unpack('f', f.read(4))[0] for i in range(count_data)]
        # print(data)
        f.close()
        return data
    except ValueError:
        print("======> Error: 输入参数应为字符串！(%s)" % file_path)
        return list()
    except IOError:
        print("======> Error: 没有找到文件或读取文件失败(%s)" % file_path)
        return list()
    except BaseException:
        return list()


def txt_reader(file_path, return_ref=0):
    try:
        with open(file_path, 'rb') as f:
            # 获取文件字符数
            # count_bytes = f.seek(0, 2)
            # print(count_bytes)
            # f.seek(0)
            # 文件版本号字节长度
            len_ver_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
            ver_str = str(f.read(len_ver_str), encoding="utf-8")
            # print(ver_str)
            # 传感器类型字节长度
            len_type_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
            type_str = str(f.read(len_type_str), encoding="utf-8")
            # print(type_str)
            # 传感器安装位置字节长度
            len_pos_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
            pos_str = str(f.read(len_pos_str))
            # print(pos_str)
            # 文件存储开始时间
            id_start_time = int.from_bytes(f.read(4), byteorder='little', signed=False)
            # print(id_start_time)
            # 传感器编号字节长度
            len_sensor_num = int.from_bytes(f.read(1), byteorder='little', signed=False)
            # print(len_sensor_num)
            # 传感器编号
            sensor_num = str(f.read(len_sensor_num), encoding="utf-8")
            # print(sensor_num)
            # 采样频率
            sample_frq = struct.unpack('f', f.read(4))[0]
            # print(sample_frq)
            # 采样精度
            sample_precision = struct.unpack('f', f.read(4))[0]
            # print(sample_precision)
            # 放大倍数
            scalor = int.from_bytes(f.read(4), byteorder='little', signed=False)
            # print(scalor)
            # 灵敏度
            sensitivity = struct.unpack('f', f.read(4))[0]
            # print(sensitivity)
            # 字符'$', 表示文件头的终结,计算出二进制字节数
            f.read(1)
            # print(str(f.read(1), encoding="utf-8"))
            bin_len = f.seek(0, 1)
            data_str = [str(data_str_i, encoding="utf-8")[0:-2].split(sep=",")
                        for data_str_i in f.readlines()]
            if type(return_ref) == int:
                return [float(data_str_i[return_ref]) for data_str_i in data_str]
            elif type(return_ref) == list:
                return [[float(data_str_i[i]) for data_str_i in data_str] for i in return_ref]
            else:
                return [float(data_str_i[0]) for data_str_i in data_str]
    except BaseException:
        if type(return_ref) == int:
            return []
        else:
            return tuple([] for i in return_ref)


def gnss_reader(file_path, return_ref=2):
    # int return_ref = 0, 1, 2 分别为x、y、z三个方向
    try:
        with open(file_path, 'r') as f:
            data_str = [data_str_i[0:-2].split(sep=",")
                        for data_str_i in f.readlines()]
            t = [datetime.datetime.strptime(data_str[i][0], "%Y-%m-%d %H:%M:%S")
                 for i in range(0, len(data_str), 2)]
            if type(return_ref) == int:
                return t, [float(data_str[i][return_ref + 2]) for i in range(0, len(data_str), 2)]
            elif type(return_ref) == list:
                return t, [[float(data_str[i][j + 2]) for i in range(0, len(data_str), 2)] for j in return_ref]
            else:
                return t, [float(data_str[i][return_ref + 2]) for i in range(0, len(data_str), 2)]
    except BaseException:
        if type(return_ref) == int:
            return [], []
        else:
            return [], tuple([] for i in return_ref)


# len1 = bin_reader(r"E:\RecycleBin~1cdd.ffs_tmp\data\新建文件夹\南京三桥数据201801\WD\温度\2018\01\22\WD010101_200009.WD")
# # len1 = bin_reader(r"I:\JSTI\数据分析组文件\江阴2021中铁桥隧升级改造样本\数据\数据\原始数据\data\WD\2021\09\01\WD060201_210000.WD")
# # len1 = bin_reader(r"E:\RecycleBin~1cdd.ffs_tmp\data\新建文件夹\南京三桥数据201801\WD\温度\2018\01\30\WD010101_150041.WD")
# print(len1)


def wind_reader(file_path, return_ref=[0, 1, 2, 3]):
    # int return_ref = 0, 1, 2 分别为x、y、z三个方向
    try:
        # with open(file_path, 'r') as f:
        f = open(file_path, 'rb')
        count = 0  # 已读取字节数
        # 获取文件字符数
        count_bytes = f.seek(0, 2)
        # print(count_bytes)
        f.seek(0)
        # 文件版本号字节长度
        len_ver_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
        ver_str = str(f.read(len_ver_str), encoding="utf-8")
        # print(ver_str)
        # 传感器类型字节长度
        len_type_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
        type_str = str(f.read(len_type_str), encoding="utf-8")
        # print(type_str)
        # 传感器安装位置字节长度
        len_pos_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
        pos_str = str(f.read(len_pos_str))
        # print(pos_str)
        # 文件存储开始时间
        id_start_time = int.from_bytes(f.read(4), byteorder='little', signed=False)
        # print(id_start_time)
        # 传感器编号字节长度
        len_sensor_num = int.from_bytes(f.read(1), byteorder='little', signed=False)
        # print(len_sensor_num)
        # 传感器编号
        sensor_num = str(f.read(len_sensor_num), encoding="utf-8")
        # print(sensor_num)
        # 采样频率
        sample_frq = struct.unpack('f', f.read(4))[0]
        # print(sample_frq)
        # 采样精度
        sample_precision = struct.unpack('f', f.read(4))[0]
        # print(sample_precision)
        # 放大倍数
        scalor = int.from_bytes(f.read(4), byteorder='little', signed=False)
        # print(scalor)
        # 灵敏度
        sensitivity = struct.unpack('f', f.read(4))[0]
        # print(sensitivity)
        # 字符'$', 表示文件头的终结
        f.read(1)
        # print(str(f.read(1), encoding="utf-8"))
        data_str = [str(data_str_i, encoding="utf-8")[0:-2].split(sep=",")
                    for data_str_i in f.readlines()]
        f.close()
        if type(return_ref) == int:
            return [float(data_str[i][return_ref]) for i in range(0, len(data_str), 1)]
        elif type(return_ref) == list:
            return ([float(data_str[i][j]) for i in range(0, len(data_str), 1)] for j in return_ref)
    except BaseException:
        f.close()
        if type(return_ref) == int:
            return []
        else:
            return tuple([] for i in return_ref)


if __name__ == "__main__":
    file_path = r"I:\JSTI\数据\江阴\江阴数据\FS\12\27\FS060101_090000.FS"
    return_ref = [0, 1, 2, 3]
    p1, p2, p3, p4 = wind_reader(file_path, return_ref)
    print(len(p1))
    print(p1[0])
