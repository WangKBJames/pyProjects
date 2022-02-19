# -*- coding: utf-8 -*-
import numpy as np


def get_range(x, data_x, frac):
    '''
    以x为中心，找着frac的比例截取数据
    :param x: 数据中心
    :param data_x: 数据x
    :param frac: 截取比例
    :return: np.array
    '''
    x_ind = np.argwhere(data_x == x)[0][0]
    half_len_w = int(np.floor((data_x.shape[0] * frac) // 2))
    len_x_list = 2 * half_len_w + 1
    if (x_ind - half_len_w) < 0:
        x_list = data_x[0:x_ind + half_len_w + 1]
        x_list = np.insert(x_list, 0, np.flipud(x_list)[0:half_len_w - x_ind])
        return x_list
    if (x_ind + half_len_w) > len(data_x):
        x_list = data_x[x_ind - half_len_w:]
        x_list = np.insert(x_list, len(x_list), np.flipud(x_list[0:len_x_list - len(x_list)]))
        return x_list
    x_list = data_x[x_ind - half_len_w:x_ind + half_len_w + 1]
    return x_list


def calFuncW(x_list):
    '''
    以w函数计算权值函数
    :param x_list: 计算权值的x序列
    :return:
    '''
    len_x_list = len(x_list)
    x_norm = np.linspace(-1, 1, len_x_list + 2)
    w = (1 - x_norm ** 2) ** 2
    return w[1:-1]


def weightRegression(x_list, y_list, w):
    '''
    计算
    :param x_list:
    :param y_list:
    :param w:
    :return:
    '''
    x2 = x_list.reshape(1, len(x_list))
    y_list_regress = x2.dot(np.linalg.inv(x2.T.dot((w * x_list).reshape(1, len(x_list))))).dot(x2.T.dot((w * y_list).reshape(1, len(x_list))))
    return y_list_regress.reshape(len(y_list))


def rlowess(data_x, data_y, frac, iter):
    '''
    鲁棒性的加权回归：
    Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”. Journal of the American Statistical Association 74 (368): 829-836.
    :param data_x:
    :param data_y:
    :param frac:
    :return:
    '''
    pass


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


if __name__ == "__main__":
    data_x = np.arange(10)
    x = 0
    frac = 0.4
    get_range(x, data_x, frac)

    pass
