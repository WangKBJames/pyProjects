from mpl_toolkits.mplot3d.axes3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

import matplotlib as mpl

import numpy as np

import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure(figsize=(10, 5))

ax = fig.gca(projection='3d')

theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)

z = np.linspace(-2, 2, 100)

r = z ** 2 + 1

x = r * np.sin(theta)

y = r * np.cos(theta)

"""

Scaling is done from here...

"""

x_scale = 10

y_scale = 1

z_scale = 3

scale = np.diag([x_scale, y_scale, z_scale, 1.0])

scale = scale * (1.0 / scale.max())

scale[3, 3] = 0.5


def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)


ax.get_proj = short_proj

"""

to here

"""

ax.plot(z, y, x, label='parametric curve')

ax.legend()

plt.show()
