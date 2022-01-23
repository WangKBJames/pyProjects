# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:48:34 2018

@author: zhen
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def expand(a, b):
    d = (b - a) * 0.1
    return a - d, b + d


if __name__ == "__main__":
    N = 1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)
    # 归一化数据
    data = StandardScaler().fit_transform(data)
    # 数据的参数
    params = ((0.2, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15))

    # 设置中文样式
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 设置颜色
    cm = matplotlib.colors.ListedColormap(list('rgbm'))
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.suptitle(u'DBSCAN聚类', fontsize=20)

    for i in range(6):
        eps, min_samples = params[i]
        # 创建密度聚类模型
        model = DBSCAN(eps=eps, min_samples=min_samples)
        # 训练模型
        model.fit(data)
        y_hat = model.labels_

        core_indices = np.zeros_like(y_hat, dtype=bool)
        core_indices[model.core_sample_indices_] = True

        y_unique = np.unique(y_hat)
        n_clusters = y_unique.size - (1 if -1 in y_hat else 0)
        # print(y_unique, '聚类簇的个数：', n_clusters)

        plt.subplot(2, 3, i + 1)
        clrs = plt.cm.Spectral(np.linspace(0, 0.8, y_unique.size))
        # print(clrs)

        x1_min, x2_min = np.min(data, axis=0)
        x1_max, x2_max = np.max(data, axis=0)
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)

        for k, clr in zip(y_unique, clrs):
            cur = (y_hat == k)
            if k == -1:
                plt.scatter(data[cur, 0], data[cur, 1], s=20, c='k')
            # 设置散点图数据
            plt.scatter(data[cur, 0], data[cur, 1], s=20, cmap=cm, edgecolors='k')
            plt.scatter(data[cur & core_indices][:, 0], data[cur & core_indices][:, 1],
                        s=20, cmap=cm, marker='o', edgecolors='k')
        # 设置x,y轴
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.grid(True)
        plt.title(u'epsilon = %.1f m = %d, 聚类数目：%d' % (eps, min_samples, n_clusters), fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
