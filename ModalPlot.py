import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 三维，两个特征
fig = plt.figure(figsize=(9, 3),dpi=80) #设置图标的大小
# ax = fig.add_subplot(111, projection='3d',) # 111的意思是把画布分为1行1列，画在第一个方格内。其实就是整个画布。
ax = fig.add_subplot(111, figsize=(8, 2.5))

# 创建样本，注意两个特征不能线性相关，否则无法用最小二乘解参数
X1 = np.arange(-4, 4, 0.1)
m = len(X1)
X2 = np.random.rand(m)*5
# print(X2)
# print(X1)

# 堆叠全1数组和X1以及X2形成样本的矩阵，倒置，用以矩阵乘法
X = np.vstack((np.full(m, 1), X1, X2)).T

# y = 15*X1 + 3 * X2 + theta0
# 自定义样本输出
Y = X1 + 3 * X2 + 3*np.random.randn(m)

# 利用标准方程(最小二乘法求解theta)
theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)),
np.transpose(X)), Y)
print(theta)

# 构造网格 meshgrid函数可以通过延伸矩阵构造多维坐标中的网格坐标。
M, N = np.meshgrid(X1, X2)

# zip函数构建一个多元祖[(x1,y1),(x2,y2)...],ravel函数将一个多维数组合并成一维数组
Z = np.array([theta[1] * d + theta[2]*p + theta[0] for d, p in zip(np.ravel(M), np.ravel(N))]).reshape(M.shape)

# 根据网格和函数构建图形 suface是一个完整的面
ax.plot_surface(M, N, Z)
ax.plot_surface
# scatter是散点图
ax.scatter(X1, X2, Y, c='r')
fig.(figsize=(8, 4), dpi=40)
# fig.set_figwidth(10)
# 设置坐标轴的名称
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()