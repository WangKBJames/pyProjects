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
    out_ind = np.argwhere(np.abs(dy) >= rate_threshould).T[0]
    # 第一次过滤
    out_single = []
    for i in out_ind:
        if i - 1 in out_ind:
            out_single.append(i)
    out_single = np.unique(out_single)
    for i in out_single:
        if i + 1 not in out_single:
            y_norm[i] = (y_norm[i-1] + y_norm[i+1])/2
        else:
            j = i
            while True:
                j += 1
                if j + 1 not in out_single:
                    y_norm[i] = y_norm[i - 1] + (y_norm[j+1] - y_norm[i - 1]) / (j - i + 2)
                    break

    # 二次过滤
    y_std3 = np.nanstd(y_norm) * 3
    out_alternative = np.argwhere(np.abs(y_norm) > y_std3).T[0]
    margin_ind = []
    # margin_ind_i = [out_alternative[0]]
    margin_ind_i = []
    y_norm_back = y_norm
    for i in range(0, len(out_alternative)):
        if i == 0:
            margin_ind_i.append(out_alternative[i])
        if i == len(out_alternative) - 1:
            margin_ind_i.append(out_alternative[i])
        if i > 0 and (out_alternative[i] - out_alternative[i - 1]) > 1:
            margin_ind_i.append(out_alternative[i])
        if i < len(out_alternative) - 1 and (out_alternative[i + 1] - out_alternative[i]) > 1 or i == (len(out_alternative) - 2):
            margin_ind_i.append(out_alternative[i])
        if len(margin_ind_i) == 2:
            margin_ind.append(margin_ind_i)
            margin_ind_i = []
    for margin_ind_i in margin_ind:
        if margin_ind_i[0] - 1 in out_ind and margin_ind_i[1] in out_ind:
            print('====>'+np.str(margin_ind_i[0])+","+np.str(margin_ind_i[1]))
            margin_i = margin_ind_i[0]
            margin_j = margin_ind_i[1]
            while True:
                if margin_i - 1 in out_ind:
                    margin_i -= 1
                elif margin_i - 2 in out_ind:
                    margin_i -= 2
                elif margin_i - 3 in out_ind:
                    margin_i -= 3
                elif margin_i - 4 in out_ind:
                    margin_i -= 4
                elif margin_i - 5 in out_ind:
                    margin_i -= 5
                elif margin_i - 6 in out_ind:
                    margin_i -= 6
                elif margin_i - 7 in out_ind:
                    margin_i -= 7
                elif margin_i - 8 in out_ind:
                    margin_i -= 8
                elif margin_i - 9 in out_ind:
                    margin_i -= 9
                elif margin_i - 10 in out_ind:
                    margin_i -= 10
                elif margin_i - 11 in out_ind:
                    margin_i -= 11
                elif margin_i - 12 in out_ind:
                    margin_i -= 12
                elif margin_i - 13 in out_ind:
                    margin_i -= 13
                elif margin_i - 14 in out_ind:
                    margin_i -= 14
                elif margin_i - 15 in out_ind:
                    margin_i -= 15
                elif margin_i - 16 in out_ind:
                    margin_i -= 16
                elif margin_i - 17 in out_ind:
                    margin_i -= 17
                elif margin_i - 18 in out_ind:
                    margin_i -= 18
                elif margin_i - 19 in out_ind:
                    margin_i -= 19
                elif margin_i - 20 in out_ind:
                    margin_i -= 20
                elif margin_i - 21 in out_ind:
                    margin_i -= 21
                elif margin_i - 22 in out_ind:
                    margin_i -= 22
                elif margin_i - 23 in out_ind:
                    margin_i -= 23
                elif margin_i - 24 in out_ind:
                    margin_i -= 24
                elif margin_i - 25 in out_ind:
                    margin_i -= 25
                elif margin_i - 26 in out_ind:
                    margin_i -= 26
                elif margin_i - 27 in out_ind:
                    margin_i -= 27
                elif margin_i - 28 in out_ind:
                    margin_i -= 28
                elif margin_i - 29 in out_ind:
                    margin_i -= 29
                elif margin_i - 30 in out_ind:
                    margin_i -= 30
                elif margin_i - 31 in out_ind:
                    margin_i -= 31
                elif margin_i - 32 in out_ind:
                    margin_i -= 32
                elif margin_i - 33 in out_ind:
                    margin_i -= 33
                elif margin_i - 34 in out_ind:
                    margin_i -= 34
                elif margin_i - 35 in out_ind:
                    margin_i -= 35
                elif margin_i - 36 in out_ind:
                    margin_i -= 36
                elif margin_i - 37 in out_ind:
                    margin_i -= 37
                elif margin_i - 38 in out_ind:
                    margin_i -= 38
                elif margin_i - 39 in out_ind:
                    margin_i -= 39
                elif margin_i - 40 in out_ind:
                    margin_i -= 40
                else:
                    break
            margin_j_p = margin_j + 1
            while True:
                if margin_j_p + 1 in out_ind:
                    margin_j_p += 1
                elif margin_j_p + 2 in out_ind:
                    margin_j_p += 2
                elif margin_j_p + 3 in out_ind:
                    margin_j_p += 3
                elif margin_j_p + 4 in out_ind:
                    margin_j_p += 4
                elif margin_j_p + 5 in out_ind:
                    margin_j_p += 5
                elif margin_j_p + 6 in out_ind:
                    margin_j_p += 6
                elif margin_j_p + 7 in out_ind:
                    margin_j_p += 7
                elif margin_j_p + 8 in out_ind:
                    margin_j_p += 8
                elif margin_j_p + 9 in out_ind:
                    margin_j_p += 9
                elif margin_j_p + 10 in out_ind:
                    margin_j_p += 10
                elif margin_j_p + 11 in out_ind:
                    margin_j_p += 11
                elif margin_j_p + 12 in out_ind:
                    margin_j_p += 12
                elif margin_j_p + 13 in out_ind:
                    margin_j_p += 13
                elif margin_j_p + 14 in out_ind:
                    margin_j_p += 14
                elif margin_j_p + 15 in out_ind:
                    margin_j_p += 15
                elif margin_j_p + 16 in out_ind:
                    margin_j_p += 16
                elif margin_j_p + 17 in out_ind:
                    margin_j_p += 17
                elif margin_j_p + 18 in out_ind:
                    margin_j_p += 18
                elif margin_j_p + 19 in out_ind:
                    margin_j_p += 19
                elif margin_j_p + 20 in out_ind:
                    margin_j_p += 20
                elif margin_j_p + 21 in out_ind:
                    margin_j_p += 21
                elif margin_j_p + 22 in out_ind:
                    margin_j_p += 22
                elif margin_j_p + 23 in out_ind:
                    margin_j_p += 23
                elif margin_j_p + 24 in out_ind:
                    margin_j_p += 24
                elif margin_j_p + 25 in out_ind:
                    margin_j_p += 25
                elif margin_j_p + 26 in out_ind:
                    margin_j_p += 26
                elif margin_j_p + 27 in out_ind:
                    margin_j_p += 27
                elif margin_j_p + 28 in out_ind:
                    margin_j_p += 28
                elif margin_j_p + 29 in out_ind:
                    margin_j_p += 29
                elif margin_j_p + 30 in out_ind:
                    margin_j_p += 30
                elif margin_j_p + 31 in out_ind:
                    margin_j_p += 31
                elif margin_j_p + 32 in out_ind:
                    margin_j_p += 32
                elif margin_j_p + 33 in out_ind:
                    margin_j_p += 33
                elif margin_j_p + 34 in out_ind:
                    margin_j_p += 34
                elif margin_j_p + 35 in out_ind:
                    margin_j_p += 35
                elif margin_j_p + 36 in out_ind:
                    margin_j_p += 36
                elif margin_j_p + 37 in out_ind:
                    margin_j_p += 37
                elif margin_j_p + 38 in out_ind:
                    margin_j_p += 38
                elif margin_j_p + 39 in out_ind:
                    margin_j_p += 39
                elif margin_j_p + 40 in out_ind:
                    margin_j_p += 40
                else:
                    margin_j = margin_j_p - 1
                    break
            out_data = y_norm[margin_i:margin_j_p]
            d_out_data = np.diff(out_data)
            d_out_data[np.argwhere(np.abs(d_out_data) > rate_threshould)] = 0
            out_data_zero = np.insert(np.cumsum(d_out_data), 0, 0.0)
            y_fit_start = y_norm[margin_i - 1]
            y_fit_end = y_norm[margin_j_p]
            y_err = (y_fit_end - y_fit_start) - (out_data_zero[-1] - out_data_zero[0])
            x_fit = np.arange(len(out_data_zero))
            d_y = x_fit * y_err/len(out_data_zero)
            out_data_back = y_fit_start + out_data_zero + d_y
            y_norm_back[margin_i:margin_j_p] = out_data_back
        elif margin_ind_i[0] - 1 in out_ind:
            margin_i = margin_ind_i[0]
            margin_j = margin_ind_i[1]
            while True:
                if margin_i - 1 in out_ind:
                    margin_i -= 1
                elif margin_i - 2 in out_ind:
                    margin_i -= 2
                elif margin_i - 3 in out_ind:
                    margin_i -= 3
                elif margin_i - 4 in out_ind:
                    margin_i -= 4
                elif margin_i - 5 in out_ind:
                    margin_i -= 5
                elif margin_i - 6 in out_ind:
                    margin_i -= 6
                elif margin_i - 7 in out_ind:
                    margin_i -= 7
                elif margin_i - 8 in out_ind:
                    margin_i -= 8
                elif margin_i - 9 in out_ind:
                    margin_i -= 9
                elif margin_i - 10 in out_ind:
                    margin_i -= 10
                elif margin_i - 11 in out_ind:
                    margin_i -= 11
                elif margin_i - 12 in out_ind:
                    margin_i -= 12
                elif margin_i - 13 in out_ind:
                    margin_i -= 13
                elif margin_i - 14 in out_ind:
                    margin_i -= 14
                elif margin_i - 15 in out_ind:
                    margin_i -= 15
                elif margin_i - 16 in out_ind:
                    margin_i -= 16
                elif margin_i - 17 in out_ind:
                    margin_i -= 17
                elif margin_i - 18 in out_ind:
                    margin_i -= 18
                elif margin_i - 19 in out_ind:
                    margin_i -= 19
                elif margin_i - 20 in out_ind:
                    margin_i -= 20
                elif margin_i - 21 in out_ind:
                    margin_i -= 21
                elif margin_i - 22 in out_ind:
                    margin_i -= 22
                elif margin_i - 23 in out_ind:
                    margin_i -= 23
                elif margin_i - 24 in out_ind:
                    margin_i -= 24
                elif margin_i - 25 in out_ind:
                    margin_i -= 25
                elif margin_i - 26 in out_ind:
                    margin_i -= 26
                elif margin_i - 27 in out_ind:
                    margin_i -= 27
                elif margin_i - 28 in out_ind:
                    margin_i -= 28
                elif margin_i - 29 in out_ind:
                    margin_i -= 29
                elif margin_i - 30 in out_ind:
                    margin_i -= 30
                elif margin_i - 31 in out_ind:
                    margin_i -= 31
                elif margin_i - 32 in out_ind:
                    margin_i -= 32
                elif margin_i - 33 in out_ind:
                    margin_i -= 33
                elif margin_i - 34 in out_ind:
                    margin_i -= 34
                elif margin_i - 35 in out_ind:
                    margin_i -= 35
                elif margin_i - 36 in out_ind:
                    margin_i -= 36
                elif margin_i - 37 in out_ind:
                    margin_i -= 37
                elif margin_i - 38 in out_ind:
                    margin_i -= 38
                elif margin_i - 39 in out_ind:
                    margin_i -= 39
                elif margin_i - 40 in out_ind:
                    margin_i -= 40
                else:
                    break
            margin_j_p = margin_j + 1
            out_data = y_norm[margin_i:margin_j_p]
            d_out_data = np.diff(out_data)
            d_out_data[np.argwhere(np.abs(d_out_data) > rate_threshould)] = 0
            out_data_zero = np.insert(np.cumsum(d_out_data), 0, 0.0)
            y_fit_start = y_norm[margin_i - 1]
            y_fit_end = y_norm[margin_j_p]
            y_err = (y_fit_end - y_fit_start) - (out_data_zero[-1] - out_data_zero[0])
            x_fit = np.arange(len(out_data_zero))
            d_y = x_fit * y_err/len(out_data_zero)
            out_data_back = y_fit_start + out_data_zero + d_y
            y_norm_back[margin_i:margin_j_p] = out_data_back
        elif margin_ind_i[1] in out_ind:
            margin_i = margin_ind_i[0]
            margin_j = margin_ind_i[1]
            margin_j_p = margin_j + 1
            while True:
                if margin_j_p + 1 in out_ind:
                    margin_j_p += 1
                elif margin_j_p + 2 in out_ind:
                    margin_j_p += 2
                elif margin_j_p + 3 in out_ind:
                    margin_j_p += 3
                elif margin_j_p + 4 in out_ind:
                    margin_j_p += 4
                elif margin_j_p + 5 in out_ind:
                    margin_j_p += 5
                elif margin_j_p + 6 in out_ind:
                    margin_j_p += 6
                elif margin_j_p + 7 in out_ind:
                    margin_j_p += 7
                elif margin_j_p + 8 in out_ind:
                    margin_j_p += 8
                elif margin_j_p + 9 in out_ind:
                    margin_j_p += 9
                elif margin_j_p + 10 in out_ind:
                    margin_j_p += 10
                elif margin_j_p + 11 in out_ind:
                    margin_j_p += 11
                elif margin_j_p + 12 in out_ind:
                    margin_j_p += 12
                elif margin_j_p + 13 in out_ind:
                    margin_j_p += 13
                elif margin_j_p + 14 in out_ind:
                    margin_j_p += 14
                elif margin_j_p + 15 in out_ind:
                    margin_j_p += 15
                elif margin_j_p + 16 in out_ind:
                    margin_j_p += 16
                elif margin_j_p + 17 in out_ind:
                    margin_j_p += 17
                elif margin_j_p + 18 in out_ind:
                    margin_j_p += 18
                elif margin_j_p + 19 in out_ind:
                    margin_j_p += 19
                elif margin_j_p + 20 in out_ind:
                    margin_j_p += 20
                elif margin_j_p + 21 in out_ind:
                    margin_j_p += 21
                elif margin_j_p + 22 in out_ind:
                    margin_j_p += 22
                elif margin_j_p + 23 in out_ind:
                    margin_j_p += 23
                elif margin_j_p + 24 in out_ind:
                    margin_j_p += 24
                elif margin_j_p + 25 in out_ind:
                    margin_j_p += 25
                elif margin_j_p + 26 in out_ind:
                    margin_j_p += 26
                elif margin_j_p + 27 in out_ind:
                    margin_j_p += 27
                elif margin_j_p + 28 in out_ind:
                    margin_j_p += 28
                elif margin_j_p + 29 in out_ind:
                    margin_j_p += 29
                elif margin_j_p + 30 in out_ind:
                    margin_j_p += 30
                elif margin_j_p + 31 in out_ind:
                    margin_j_p += 31
                elif margin_j_p + 32 in out_ind:
                    margin_j_p += 32
                elif margin_j_p + 33 in out_ind:
                    margin_j_p += 33
                elif margin_j_p + 34 in out_ind:
                    margin_j_p += 34
                elif margin_j_p + 35 in out_ind:
                    margin_j_p += 35
                elif margin_j_p + 36 in out_ind:
                    margin_j_p += 36
                elif margin_j_p + 37 in out_ind:
                    margin_j_p += 37
                elif margin_j_p + 38 in out_ind:
                    margin_j_p += 38
                elif margin_j_p + 39 in out_ind:
                    margin_j_p += 39
                elif margin_j_p + 40 in out_ind:
                    margin_j_p += 40
                else:
                    margin_j = margin_j_p - 1
                    break
            out_data = y_norm[margin_i:margin_j_p]
            d_out_data = np.diff(out_data)
            d_out_data[np.argwhere(np.abs(d_out_data) > rate_threshould)] = 0
            out_data_zero = np.insert(np.cumsum(d_out_data), 0, 0.0)
            y_fit_start = y_norm[margin_i - 1]
            y_fit_end = y_norm[margin_j_p]
            y_err = (y_fit_end - y_fit_start) - (out_data_zero[-1] - out_data_zero[0])
            x_fit = np.arange(len(out_data_zero))
            d_y = x_fit * y_err / len(out_data_zero)
            out_data_back = y_fit_start + out_data_zero + d_y
            y_norm_back[margin_i:margin_j_p] = out_data_back
    return y_norm_back


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

    if False:
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

    if True:
        from dataReader import gnss_data

        main_path = r"D:\pytestdata"
        sensor_num = "BD080102"
        t_start_list = [2021, 8, 21, 0, 0, 0]
        t_end_list = [2021, 8, 30, 23, 0, 0]
        t_list, data = gnss_data(main_path, sensor_num, t_start_list, t_end_list, return_ref=[0, 1, 2], sample_frq=1)
        nd = np.array(data[2], dtype='float') * 100
        nd = nd - np.nanmean(nd)
        nd[np.isnan(nd)] = np.nanmean(nd)
        x = np.arange(len(nd))
        nd_hat, w_list = rloess(x, nd, frac=0.05, step=2000, iters=4)
        rate_threshould = 10.0
        nd_back = isoutlier(nd, nd_hat, rate_threshould)


        # dnd_represent = []
        # for i in x:
        #     if np.abs(dnd[i]) > 10:
        #         dnd_represent.append(dnd[i])
        #     else:
        #         dnd_represent.append(0.0)
        # dnd_represent = np.array(dnd_represent).cumsum()

        fig = plt.figure(figsize=(12, 12))  # 定义图并设置画板尺寸
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        # fig.tight_layout()                                                    # 调整整体空白
        # plt.subplots_adjust(bottom=0.25, top=0.94, left=0.08, right=0.94, wspace=0.36, hspace=0.5)
        ax1 = fig.add_subplot(111)  # 定义子图
        # plt.xticks(rotation=90)
        ax1.plot(x, nd-nd_hat, 'b')
        ax1.plot(x, nd_back, 'r')
        # ax1.plot(x, nd_hat)
        # ax.plot(data_x, data_y_hat2.T[1])
        # ax2 = fig.add_subplot(212)  # 定义子图
        # # plt.xticks(rotation=90)
        # ax2.plot(data_x, data_y - data_y_hat, 'b')
        plt.show()
