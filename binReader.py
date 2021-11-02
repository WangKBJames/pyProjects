# -*- coding: utf-8 -*-
import struct


def bin_reader(file_path):
    try:
        f = open(file_path, 'rb')
        count = 0  # 已读取字节数
        # 获取文件字符数
        count_bytes = f.seek(0, 2)
        print(count_bytes)
        f.seek(0)
        # 文件版本号字节长度
        len_ver_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
        ver_str = str(f.read(len_ver_str), encoding="utf-8")
        print(ver_str)
        # 传感器类型字节长度
        len_type_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
        type_str = str(f.read(len_type_str), encoding="utf-8")
        print(type_str)
        # 传感器安装位置字节长度
        len_pos_str = int.from_bytes(f.read(1), byteorder='little', signed=False)
        pos_str = str(f.read(len_pos_str))
        print(pos_str)
        # 文件存储开始时间
        id_start_time = int.from_bytes(f.read(4), byteorder='little', signed=False)
        print(id_start_time)
        # 传感器编号字节长度
        len_sensor_num = int.from_bytes(f.read(1), byteorder='little', signed=False)
        print(len_sensor_num)
        # 传感器编号
        sensor_num = str(f.read(len_sensor_num), encoding="utf-8")
        print(sensor_num)
        # 采样频率
        sample_frq = struct.unpack('f', f.read(4))[0]
        print(sample_frq)
        # 采样精度
        sample_precision = struct.unpack('f', f.read(4))[0]
        print(sample_precision)
        # 放大倍数
        scalor = int.from_bytes(f.read(4), byteorder='little', signed=False)
        print(scalor)
        # 灵敏度
        sensitivity = struct.unpack('f', f.read(4))[0]
        print(sensitivity)
        # 字符'$', 表示文件头的终结
        print(str(f.read(1), encoding="utf-8"))
        data = list()
        # 数据数量
        count_data = int((count_bytes - f.seek(0, 1)) / 4)
        print(count_data)
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

#
# len1 = bin_reader(r"E:\RecycleBin~1cdd.ffs_tmp\data\新建文件夹\南京三桥数据201801\WD\温度\2018\01\22\WD010101_200009.WD")
# # len1 = bin_reader(r"I:\JSTI\数据分析组文件\江阴2021中铁桥隧升级改造样本\数据\数据\原始数据\data\WD\2021\09\01\WD060201_210000.WD")
# # len1 = bin_reader(r"E:\RecycleBin~1cdd.ffs_tmp\data\新建文件夹\南京三桥数据201801\WD\温度\2018\01\30\WD010101_150041.WD")
# print(len1)
