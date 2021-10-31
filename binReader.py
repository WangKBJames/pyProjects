# -*- coding: utf-8 -*-
import struct


def bin_reader(file_path):
    f = open(file_path, 'rb')
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
    # 放大倍数
    scalor = int.from_bytes(f.read(4), byteorder='little', signed=False)
    print(scalor)
    # 灵敏度
    sensitivity = struct.unpack('f', f.read(4))[0]
    print(sensitivity)
    # 字符'$', 表示文件头的终结
    print(str(f.read(1), encoding="utf-8"))

    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)
    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)
    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)
    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)
    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)
    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)
    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)
    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)
    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)
    data1 = struct.unpack('f', f.read(4))[0]
    print(data1)





    # b=fread(fileID,1,'int8');         % 传感器编号字节长度
    # b=fread(fileID,b,'*char')';      % 传感器编号
    # b = fread(fileID,1,'float');           % 采样频率
    # b=fread(fileID,1,'int32');        % 采样精度
    # b=fread(fileID,1,'int32');         % 放大倍数
    # b = fread(fileID,1,'float');           % 灵敏度
    # b = fread(fileID,1,'*char');            % 字符'$',表示文件头的终结
    # data = fread(fileID,inf,'float',0);


    return len_ver_str


len1 = bin_reader(r"I:\JSTI\数据分析组文件\江阴位移\wy202004\2020\04\13\WY020101_000000.WY")
# len1 = bin_reader(r"I:\JSTI\数据分析组文件\江阴2021中铁桥隧升级改造样本\数据\数据\原始数据\data\WD\2021\09\01\WD060201_210000.WD")
# print(len1)
