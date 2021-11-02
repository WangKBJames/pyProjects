import re
from timeParser import time_parser, time_list
import numpy as np

a = np.array([1, 2, 1, 3, 2])
a.fill()
a = r"I:\JSTI\数据分析组文件\江阴2021中铁桥隧升级改造样本\数据\数据\原始数据\data\WD\2021\09\01\WD060201_210000.WD"
t = time_parser(a)
tl = time_list(t, 10, 1)
print(tl)
# # match_obj = re.match(r'.*_(\d{6})\..*', a, re.M | re.I)
# match_obj = re.match(r'.*\\(\d{4})\\(\d{2})\\(\d{2})\\.*_(\d{6})\..*', a, re.M | re.I)
# print(match_obj.group(1))
# print(match_obj.group(2))
# print(match_obj.group(3))
# print(type(match_obj.group(4)))
np.arange()