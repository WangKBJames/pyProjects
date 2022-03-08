# -*- coding: utf-8 -*-
import numpy as np

main_path = r".\tmp_field\\"


def tmp_gradient(tmp_1, tmp_2, dis=5):
    '''
    温度梯度和温度极值分析
    :param tmp_1: 下拉菜单1选择的结构温度传感器数据 图1 y轴数据1 x轴为时间，折线图
    :param tmp_2: 下拉菜单2选择的结构温度传感器数据 图1 y轴数据2
    :return:list[][]
    list[0]: grad: float[],  温度梯度序列，图2 y 轴（温度梯度：单位：℃），x轴为时间
    list[1]: doc_str: string, 文本区域 说明文字
    '''

    if type(tmp_1) is not np.ndarray:
        tmp_1 = np.array(tmp_1, dtype='float')
    if type(tmp_2) is not np.ndarray:
        tmp_2 = np.array(tmp_2, dtype='float')
    grad = (tmp_2 - tmp_1) / dis
    tmp_mean = (np.nanmean(tmp_1) + np.nanmean(tmp_2)) / 2
    tmp_max = np.max([np.nanmax(tmp_1), np.nanmax(tmp_2)])
    tmp_min = np.min([np.nanmin(tmp_1), np.nanmin(tmp_2)])
    d_tmp_max = np.max([np.nanmax(tmp_1) - np.nanmin(tmp_1), np.nanmax(tmp_2) - np.nanmin(tmp_2)])
    grad_max = np.max(np.abs(grad))
    doc_str = "温度均值：{0:.2f}℃\n最高温度：{1:.2f}℃\n最低温度：{2:.2f}℃\n最大温差：" \
              "{3:.2f}℃\n最大温度梯度：{4:.2f}℃/m".format(tmp_mean, tmp_max, tmp_min,
                                                   d_tmp_max, grad_max)
    return [grad, doc_str]


def process():
    data_1 = np.loadtxt(main_path + r"input\shuju1.txt", dtype='float')
    data_2 = np.loadtxt(main_path + r"input\shuju2.txt", dtype='float')
    grad, doc_str = tmp_gradient(data_1, data_2, dis=5)
    np.savetxt(main_path + r"output\fig2_y_1.txt", grad)
    with open(main_path + r"output\shuoming.txt", "w", encoding="utf-8") as f:
        f.write(doc_str)
    return


if __name__ == "__main__":
    tmp_1 = np.random.randn(40)
    tmp_2 = np.random.randn(40)
    # grad, doc_str = tmp_gradient(tmp_1, tmp_2, dis=5)

    np.savetxt(main_path + r"input\shuju1.txt", tmp_1)
    np.savetxt(main_path + r"input\shuju2.txt", tmp_2)
    process()
