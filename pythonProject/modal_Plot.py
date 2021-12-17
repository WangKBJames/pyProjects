from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt

x = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
y = [-1, 1]
z = [[0, 0.7, 1, 0.72, 0, -0.69, -1, -0.67, 0], [0, 0.7, 1, 0.72, 0, -0.69, -1, -0.67, 0]]
z = np.array(z)
x, y = np.meshgrid(x, y)
fig = plt.figure(figsize=(9, 3), dpi=80)
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_axes([0.15, 0.1, 0.7, 0.3],projection='3d')
ax.set_position([0.05, 0.05, 0.9, 0.9])
ax.set_xticks(np.linspace(0, 1, 9))
ax.set_xlim([0,1])
ax.set_ylim([-1,1])
# ax.xlim()

# Plot a basic wireframe.
ax.plot_wireframe(x, y, z, rstride=10, cstride=2, color='#dc4105')
ax.view_init(45,250)
plt.show()
