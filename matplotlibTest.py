import numpy as np
import matplotlib.pyplot as plt

# data = np.arange(10)
# plt.plot(data)
# plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
plt.plot(np.random.randn(50).cumsum(), 'r:')
plt.show()
print(type(np.random.randn(50)))
