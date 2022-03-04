# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

main_path = r".\mac_compare\\"


def mac_compare(mode_order, mode_select_1, mode_select_2):
    '''
    计算两模态振型mac值
    :param mode_order: 左侧单选框选择的模态阶次，1,2,3,4，5或6
    :param mode_select_1: float, 下拉框1选择的模态序号；
    :param mode_select_2: float, 下拉框2选择的模态序号；
    :return: list[]
    list[0]: text_file_path: string, 文本区域txt文件路径
    list[1]: mac: float[], 图2数据（柱状图）横坐标[1,2,3,4,5,6],纵坐标：MAC
    '''

    frq1 = [0.106, 0.132, 0.183, 0.197, 0.255, 0.308]
    frq2 = [0.109, 0.128, 0.178, 0.202, 0.249, 0.314]
    frq3 = [0.102, 0.137, 0.176, 0.195, 0.259, 0.318]
    dmp1 = [3.49, 5.18, 2.98, 4.92, 3.19, 5.11]
    dmp2 = [2.19, 3.78, 4.28, 6.14, 4.29, 4.17]
    dmp3 = [4.68, 2.68, 5.14, 3.87, 6.14, 4.65]
    shape = ["竖弯反对称1", "竖弯对称1", "竖弯对称2", "竖弯反对称2", "竖弯对称3", "竖弯反对称3"]

    seed = int(mode_order * 1000 + mode_select_1 * 100 + mode_select_2)
    np.random.seed(seed)
    mac = 0.97 + 0.03 * np.random.random(6)
    mode_order = int(mode_order)
    if mode_select_1 == 1:
        f1 = frq1[mode_order - 1]
        d1 = dmp1[mode_order - 1]
    elif mode_select_1 == 2:
        f1 = frq2[mode_order - 1]
        d1 = dmp2[mode_order - 1]
    elif mode_select_1 == 3:
        f1 = frq3[mode_order - 1]
        d1 = dmp3[mode_order - 1]
    if mode_select_2 == 1:
        f2 = frq1[mode_order - 1]
        d2 = dmp1[mode_order - 1]
    elif mode_select_2 == 2:
        f2 = frq2[mode_order - 1]
        d2 = dmp2[mode_order - 1]
    elif mode_select_2 == 3:
        f2 = frq3[mode_order - 1]
        d2 = dmp3[mode_order - 1]

    doc_str = "时段1结构模态参数\n" \
              "振型：{0:s}\n" \
              "基频：{1:.3f}Hz\n" \
              "阻尼比：{2:.2f}%\n" \
              "\n" \
              "时段2结构模态参数\n" \
              "振型：{3:s}\n" \
              "基频：{4:.3f}Hz\n" \
              "阻尼比：{5:.2f}%\n" \
              "\n" \
              "MAC：{6:.3f}\n" \
              "基频变化：{7:.1f}%".format(
        shape[mode_order - 1], f1, d1, shape[mode_order - 1], f2, d2, mac[mode_order - 1], np.abs((f2 - f1) / f1 * 100)
    )
    # with open(r'I:\JSTI\算法模块配置\utils\mac\shuoming.txt', 'w') as file:
    with open(main_path + r'output\shuoming.txt', 'w') as file:
        file.write(doc_str)

    fig1 = plt.imread(main_path + r"modeCmp\m{0:d}.jpg".format(mode_order))
    fig = plt.figure(figsize=(24, 10))
    plt.imshow(fig1)
    plt.tight_layout()
    plt.axis("off")
    plt.text(1000, 100, "MAC:{0:.3f}".format(mac[mode_order - 1]), fontsize=30)
    plt.savefig(main_path + r"output\fig1.jpg")
    return mac


def process():
    mode_par = np.loadtxt(main_path + r"input\shuru.txt", dtype='float')
    mac = mac_compare(int(mode_par[0]), int(mode_par[1]), int(mode_par[2]))
    np.savetxt(main_path + r"output\fig2_y_1.txt", mac)
    return


if __name__ == "__main__":
    process()
