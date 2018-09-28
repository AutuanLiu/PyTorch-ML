"""
Email: autuanliu@163.com
Date: 2018/9/27
Ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image, 返回的结果是一个 numpy 数组
fp = 'images/dancing.jpg'
img = cv2.imread(fp, 1)
print(img, type(img), img.shape)

# 显示图片 esc
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存图片
cv2.imwrite('images/mess.png', img)

# 访问属性
print(img.shape, img.dtype, img.size)

b = img[280:340, 330:390] * 2.5
img[273:333, 100:160] = b
plt.imshow(img)
plt.show()

# Splitting and Merging Image Channels
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))