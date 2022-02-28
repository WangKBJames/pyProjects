# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as spi


# def get_range(x, data_x, frac):
#     '''
#     以x为中心，找着frac的比例截取数据
#     :param x: 数据中心
#     :param data_x: 数据x
#     :param frac: 截取比例
#     :return: np.array
#     '''
#     x_ind = np.argwhere(data_x == x)[0][0]
#     half_len_w = int(np.floor((data_x.shape[0] * frac) // 2))
#     len_x_list = 2 * half_len_w + 1
#     if (x_ind - half_len_w) < 0:
#         x_list = data_x[0:x_ind + half_len_w + 1]
#         x_list = np.insert(x_list, 0, np.flipud(x_list)[0:half_len_w - x_ind])
#         return x_list
#     if (x_ind + half_len_w) > len(data_x):
#         x_list = data_x[x_ind - half_len_w:]
#         x_list = np.insert(x_list, len(x_list), np.flipud(x_list[0:len_x_list - len(x_list)]))
#         return x_list
#     x_list = data_x[x_ind - half_len_w:x_ind + half_len_w + 1]
#     return x_list


def get_range(x, data_x, frac):
    '''
    以x为中心，找着frac的比例截取数据，端部数据取同等长度
    :param x: 数据中心
    :param data_x: 数据x
    :param frac: 截取比例
    :return: np.array
    '''
    x_ind = np.argwhere(data_x == x)[0][0]
    half_len_w = int(np.floor((data_x.shape[0] * frac) / 2))
    len_x_list = 2 * half_len_w + 1
    if (x_ind - half_len_w) < 0:
        x_list = data_x[0:len_x_list]
        x_loc = x_ind
    elif (x_ind + half_len_w) >= len(data_x):
        x_list = data_x[-len_x_list:]
        x_loc = x_ind - len(data_x)
    else:
        x_list = data_x[x_ind - half_len_w:x_ind + half_len_w + 1]
        x_loc = half_len_w
    return x_list, x_loc


def calFuncW(x_list, x_loc):
    '''
    以w函数计算权值函数
    :param x_list: 计算权值的x序列
    :return:
    '''
    len_x_list = len(x_list)
    if 2 * x_loc + 1 == len_x_list:
        x_norm = np.linspace(-1, 1, len_x_list + 2)
        w = (1 - x_norm ** 2) ** 2
        w = w[1:-1]
    elif x_loc >= 0:
        x_norm = np.linspace(-1, 1, (len_x_list - x_loc - 1) * 2 + 3)
        w = (1 - x_norm ** 2) ** 2
        w = w[-len_x_list - 1:-1]
    else:
        x_norm = np.linspace(-1, 1, (len_x_list + x_loc) * 2 + 3)
        w = (1 - x_norm ** 2) ** 2
        w = w[1:len_x_list + 1]
    return w


def weightRegression(x_list, y_list, w, fitfunc="T"):
    '''
    权重回归分析
    :param x_list:
    :param y_list:
    :param w:
    :return:
    '''
    # x2 = x_list.reshape(1, len(x_list))
    y2 = y_list.reshape(1, len(x_list))
    w2 = w.reshape(1, len(x_list))
    # y_list_regress = x2.dot(np.linalg.inv(x2.T.dot((w * x_list).reshape(1, len(x_list))))).dot(x2.T.dot((w * y_list).reshape(1, len(x_list))))
    if fitfunc == "B":
        x2 = np.ones([2, len(x_list)])
        x2[0] = x_list
        y_list_regress = x2.T.dot(np.linalg.inv(x2.dot((w2 * x2).T))).dot(x2.dot((w2 * y2).T))
    elif fitfunc == "T":
        x2 = np.ones([3, len(x_list)])
        x2[0] = x_list ** 2
        x2[1] = x_list
        y_list_regress = x2.T.dot(np.linalg.inv(x2.dot((w2 * x2).T))).dot(x2.dot((w2 * y2).T))
    return y_list_regress.reshape(len(y_list))


def cal_new_weight(y_hat, y_list, w, wfunc="B"):
    '''
    计算局部回归调整后权重
    :param y_hat: 局部回归后输出数据
    :param data_y: 原始数据
    :param func: string, "B"二次权重函数，"w"三次权重函数
    :return:
    '''
    err = y_list - y_hat
    s = np.nanmedian(np.abs(err))
    err_norm = err / 6 / s
    if wfunc == "B":
        delta_k = (1 - err_norm ** 2) ** 2
    elif wfunc == "W":
        delta_k = (1 - np.abs(err_norm) ** 3) ** 3
    delta_k[abs(err_norm) > 1] = 0
    new_w = delta_k * w
    return new_w


def rlowess(data_x, data_y, frac, iters=2):
    '''
    鲁棒性的加权回归：
    Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”. Journal of the American Statistical Association 74 (368): 829-836.
    :param data_x:
    :param data_y:
    :param frac:
    :return:
    '''
    data_y_hat = np.ones_like(data_y)
    half_len_w = int(np.floor((data_x.shape[0] * frac) // 2))
    for x in data_x:
        x_list, x_loc = get_range(x, data_x, frac)
        new_w = calFuncW(x_list, x_loc)
        y_hat = weightRegression(x_list, data_y[x_list], new_w)
        for it in range(iters):
            new_w = cal_new_weight(y_hat, data_y[x_list], new_w, wfunc="B")
            y_hat = weightRegression(x_list, data_y[x_list], new_w, "B")
        data_y_hat[x] = y_hat[x_loc]
    return data_y_hat


def rloess(data_x, data_y, frac, step=1, iters=2):
    '''
    鲁棒性的加权回归：
    Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”. Journal of the American Statistical Association 74 (368): 829-836.
    :param data_x:
    :param data_y:
    :param frac:
    :param step:
    :param iters:
    :return:
    '''
    # data_y_hat = np.ones_like(data_y)
    half_len_w = int(np.floor((data_x.shape[0] * frac) // 2))
    data_x_step = data_x[0::step]
    if data_x_step[-1] != data_x[-1]:
        data_x_step = np.append(data_x_step, data_x[-1])
    data_y_hat_step = np.random.random(len(data_x_step))
    w_list = np.random.random(len(data_x_step))
    for x in range(len(data_x_step)):
        x_list, x_loc = get_range(data_x_step[x], data_x, frac)
        new_w = calFuncW(x_list, x_loc)
        y_hat = weightRegression(x_list, data_y[x_list], new_w)
        for it in range(iters):
            new_w = cal_new_weight(y_hat, data_y[x_list], new_w, wfunc="B")
            y_hat = weightRegression(x_list, data_y[x_list], new_w)
        data_y_hat_step[x] = y_hat[x_loc]
        w_list[x] = new_w[x_loc]
    data_y_hat_rep = spi.splrep(data_x_step, data_y_hat_step, k=2)
    data_y_hat = spi.splev(data_x, data_y_hat_rep)
    return data_y_hat, w_list


def lwlr(testPoint, xArr, yArr, tao=1.0):
    """
    :param xArr: 训练集x矩阵，对于时序拟合问题，xArr即有序自然数列
    :param yArr: 训练集输出y矩阵，对于时序拟合问题，yArr即时序数列
    :param tao: 衰减因子，取值越小，周围样本权重随着距离增大而减得越快，即易过拟合，一般取值0.001-1
    :return:返回模型权重矩阵w，和样本testPoint的输出testPoint * w
    没有用到frac参数，单词迭代（未根据局部加权回归误差，对样本权重进行更新，进行下一次迭代）
    """
    # 样本数量m
    m = np.shape(xArr)[0]
    # 初始化权重矩阵weights
    weights = np.mat(np.eye((m)))
    # 更新测试点testPoint 周围的样本权重矩阵weights
    for j in range(m):
        diffMat = testPoint - xArr[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * tao ** 2))
    xTx = xArr.T * (weights * xArr)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    #
    w = xTx.I * (xArr.T * (weights * yArr))  # normal equation
    return testPoint * w


def lwlrTest(testArr, xArr, yArr, tao=1.0):
    """
    :param testArr: 测试集矩阵
    :param xArr: 训练集矩阵
    :param yArr: 训练集输出
    :param tao: 衰减因子
    :return: 测试集矩阵testArr的对应输出
    """
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, tao)
    return yHat


def isoutlier(data_y, data_y_hat, rate_threshould):
    '''

    :param data_y:
    :param data_y_hat:
    :param rate_threshould:
    :return:
    '''
    y_norm = data_y - data_y_hat
    dy = np.diff(y_norm)
    dy_mid = np.nanmedian(np.abs(dy))
    out_ind = np.argwhere(abs(dy) >= rate_threshould).T[0]
    out_flag = np.array([np.abs(y_norm[i]) > np.abs(y_norm[i-1]) for i in range(len(out_ind))])


    y_std = np.nanstd(y_norm)
    out_ind = np.argwhere(np.abs(y_norm) > 3*y_std).T[0]
    return


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    if False:
        # data_x = np.arange(100)
        # x = 30
        # frac = 0.4
        # x_list = get_range(x, data_x, frac)
        # w = calFuncW(x_list)
        # data_y = np.random.randn(100)
        # y_hat = weightRegression(x_list, data_y[x_list], w)
        # new_w = cal_new_weight(y_hat, data_y[x_list], w, wfunc="B")
        data_x = np.arange(9501)
        data_y = np.sin(0.001 * data_x) + np.random.randn(9501) * 0.1
        data_y[3000:3500] = data_y[3000:3500] + 0.5
        data_y_hat, w_list = rloess(data_x, data_y, frac=0.5, step=1, iters=4)
        # data_y_hat2 = sm.nonparametric.lowess(data_y, data_x, frac=0.2, it=10, delta=0)

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fig = plt.figure(figsize=(12, 12))  # 定义图并设置画板尺寸
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        # fig.tight_layout()                                                    # 调整整体空白
        # plt.subplots_adjust(bottom=0.25, top=0.94, left=0.08, right=0.94, wspace=0.36, hspace=0.5)
        ax1 = fig.add_subplot(211)  # 定义子图
        # plt.xticks(rotation=90)
        ax1.plot(data_x, data_y, 'b')
        ax1.plot(data_x, data_y_hat)
        # ax.plot(data_x, data_y_hat2.T[1])
        ax2 = fig.add_subplot(212)  # 定义子图
        # plt.xticks(rotation=90)
        ax2.plot(data_x, data_y - data_y_hat, 'b')
        plt.show()

    if True:
        from dataReader import gnss_data

        main_path = r"D:\pytestdata"
        sensor_num = "BD080101"
        t_start_list = [2021, 8, 21, 0, 0, 0]
        t_end_list = [2021, 8, 30, 23, 0, 0]
        t_list, data = gnss_data(main_path, sensor_num, t_start_list, t_end_list, return_ref=[0, 1, 2], sample_frq=1)
        nd = np.array(data[2], dtype='float') * 100
        nd = nd - np.nanmean(nd)
        nd[np.isnan(nd)] = np.nanmean(nd)
        x = np.arange(len(nd))
        nd_hat, w_list = rloess(x, nd, frac=0.05, step=2000, iters=4)
        fig = plt.figure(figsize=(12, 12))  # 定义图并设置画板尺寸
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        # fig.tight_layout()                                                    # 调整整体空白
        # plt.subplots_adjust(bottom=0.25, top=0.94, left=0.08, right=0.94, wspace=0.36, hspace=0.5)
        ax1 = fig.add_subplot(111)  # 定义子图
        # plt.xticks(rotation=90)
        ax1.plot(x, nd, 'b')
        ax1.plot(x, nd_hat)
        # ax.plot(data_x, data_y_hat2.T[1])
        # ax2 = fig.add_subplot(212)  # 定义子图
        # # plt.xticks(rotation=90)
        # ax2.plot(data_x, data_y - data_y_hat, 'b')
        plt.show()
