import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("./testPics/JiangYinChuanZhi1.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# plt.figure()
# plt.hist(gray.ravel(), 256)
# plt.show()
# cv2.imshow("figure", img)
# cv2.waitKey(0)

dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

