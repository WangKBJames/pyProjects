# -*- coding: utf-8 -*-
import numpy as np

main_path = r".\vehicle_load\\"


def vehicle_flow(time_num):
    '''
    计算车流量直方图
    :param time_num: float，车流量计算开始时间日期时间戳，如7538584
    :return: list[][]
    list[0]: flow_num: float[], 车流量序列，图1 y坐标，柱状图，x轴为日期，直方图
    list[1]: car_truck_ratio: float[], 客货比序列，图2 y坐标，柱状图，x轴为日期，直方图
    list[2]: axial_weight: float[], 车轴重序列，图3 y坐标，单位t，柱状图，x轴为日期， 直方图
    list[3]: dist_x: float[], 车轴重分布，图4 x轴，单位t， 直方图
    list[4]: dist_y: float[], 某一重量车的数量， 图4 y轴，直方图
    list[5]: doc_str: string, 文本区域 说明文字
    '''

    flow_num = [2.5000000000000000e+03,
                2.1000000000000000e+03,
                3.2000000000000000e+03,
                3.1000000000000000e+03,
                2.6000000000000000e+03,
                2.9500000000000000e+03,
                4.0500000000000000e+03,
                6.8000000000000000e+03,
                7.2000000000000000e+03,
                6.6000000000000000e+03,
                5.9000000000000000e+03,
                4.7000000000000000e+03,
                3.1800000000000000e+03,
                5.9400000000000000e+03,
                6.2400000000000000e+03,
                3.9400000000000000e+03,
                4.8200000000000000e+03,
                8.9200000000000000e+03,
                8.1200000000000000e+03,
                8.7200000000000000e+03,
                6.8000000000000000e+03,
                5.9000000000000000e+03,
                4.1200000000000000e+03,
                2.8100000000000000e+03]

    car_truck_ratio = [5.3294934483670247e+00,
                       3.7782341762916958e+00,
                       4.3054898234588430e+00,
                       5.6984249992703102e+00,
                       3.6075869686066269e+00,
                       4.1893674720720115e+00,
                       5.9359888835598795e+00,
                       5.5319347626644921e+00,
                       5.5887154155165941e+00,
                       5.6111436339121923e+00,
                       4.9537312633898951e+00,
                       5.5391626361462887e+00,
                       5.8381465146908971e+00,
                       3.3809800903428759e+00,
                       5.1111811902461435e+00,
                       4.5547501870935392e+00,
                       4.1407644230818139e+00,
                       3.4927198884005075e+00,
                       5.6503081325879547e+00,
                       5.0184565097837446e+00,
                       4.1817534546710391e+00,
                       6.0833336710092372e+00,
                       4.5244631165557623e+00,
                       5.3453650062054123e+00]

    axial_weight = [5.5538161891035652e+03,
                    4.8822927068496474e+03,
                    6.2285343692282631e+03,
                    6.8816632313393839e+03,
                    5.8236743399456172e+03,
                    6.5740242899564037e+03,
                    8.4647922830722528e+03,
                    1.4068632053940099e+04,
                    1.5568292142902583e+04,
                    1.3694091805754131e+04,
                    1.2175670430978756e+04,
                    9.6065725110298990e+03,
                    6.3459354484563710e+03,
                    1.2800098018297103e+04,
                    1.3228719751118111e+04,
                    7.8911381962343557e+03,
                    1.0496349505705493e+04,
                    1.8435382113662949e+04,
                    1.7357580389339048e+04,
                    1.8327221388746093e+04,
                    1.4212586068152883e+04,
                    1.2414985210510167e+04,
                    8.3382969146229334e+03,
                    5.4265198153360752e+03]

    dist_y = [1.1614000000000000e+04,
              1.3413000000000000e+04,
              1.4417000000000000e+04,
              1.3942000000000000e+04,
              1.3743000000000000e+04,
              1.0147000000000000e+04,
              7.5860000000000000e+03,
              5.5180000000000000e+03,
              3.1630000000000000e+03,
              1.7740000000000000e+03,
              9.8700000000000000e+02,
              4.3900000000000000e+02,
              2.9800000000000000e+02,
              4.1000000000000000e+01,
              4.1000000000000000e+01,
              1.1000000000000000e+01,
              1.0000000000000000e+01,
              1.0000000000000000e+01,
              7.0000000000000000e+00]

    dist_x = [3.8866830894551851e-01,
              9.8886005911172825e-01,
              1.5890518092779371e+00,
              2.1892435594441459e+00,
              2.7894353096103548e+00,
              3.3896270597765654e+00,
              3.9898188099427743e+00,
              4.5900105601089827e+00,
              5.1902023102751915e+00,
              5.7903940604414004e+00,
              6.3905858106076110e+00,
              6.9907775607738216e+00,
              7.5909693109400305e+00,
              8.1911610611062393e+00,
              8.7913528112724499e+00,
              9.3915445614386606e+00,
              9.9917363116048676e+00,
              1.0915445614386606e+01,
              1.1917363116048676e+01]

    np.random.seed(int(time_num))
    flow_num_r = 0.85 + 0.3 * np.random.random(len(flow_num))
    flow_num = np.array(flow_num) * flow_num_r

    np.random.seed(int(time_num + 100))
    car_truck_ratio_r = 0.9 + 0.2 * np.random.random(len(car_truck_ratio))
    car_truck_ratio = np.array(car_truck_ratio) * car_truck_ratio_r

    np.random.seed(int(time_num + 200))
    axial_weight_r = 0.8 + 0.4 * np.random.random(len(axial_weight))
    axial_weight = np.array(axial_weight) * axial_weight_r

    np.random.seed(int(time_num + 300))
    dist_y_r = 0.9 + 0.2 * np.random.random(len(dist_y))
    dist_y = np.array(dist_y) * dist_y_r

    count = np.floor(np.sum(flow_num))
    up_down_ratio = 0.85 + 0.3 * np.random.random()
    ct_ratio = np.mean(car_truck_ratio)
    count_car = np.floor(count * ct_ratio / (ct_ratio + 1))
    count_truck = count - count_car
    count_over_weight = 0.0
    weight_max = 24.0 + 15 * np.random.random()

    doc_str = "车辆荷载统计：\n" \
              "总车流量：{0:d}辆\n" \
              "上、下行车流量比：{1:.2f}\n" \
              "客车流量：{2:d}辆\n" \
              "货车流量：{3:d}辆\n" \
              "客货比：{4:.2f}\n" \
              "超载车辆数：{5:d}辆\n" \
              "最重车辆：{6:.2f}t".format(int(count), up_down_ratio, int(count_car), int(count_truck), ct_ratio,
                                     int(count_over_weight), weight_max)
    return [flow_num.tolist(), car_truck_ratio.tolist(), axial_weight.tolist(), dist_x, dist_y.tolist(), doc_str]


def process():
    time_num = np.loadtxt(main_path + r"input\par.txt", dtype='float')
    flow_num, car_truck_ratio, axial_weight, dist_x, dist_y, doc_str = vehicle_flow(time_num)
    np.savetxt(main_path + r"output\fig1_y_1.txt", flow_num)
    np.savetxt(main_path + r"output\fig2_y_1.txt", car_truck_ratio)
    np.savetxt(main_path + r"output\fig3_y_1.txt", axial_weight)
    np.savetxt(main_path + r"output\fig4_x.txt", dist_x)
    np.savetxt(main_path + r"output\fig4_y_1.txt", dist_y)
    with open(main_path + r"output\shuoming.txt", "w", encoding="utf-8") as f:
        f.write(doc_str)
    return


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # flow_num, car_truck_ratio, axial_weight, dist_x, dist_y, doc_str = vehicle_flow(322344)
    # print(doc_str)
    process()
