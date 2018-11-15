"""
Email: autuanliu@163.com
Date: 2018/9/27
Ref: https://pillow-zh-cn.readthedocs.io/zh_CN/latest/handbook/tutorial.html
https://www.jianshu.com/p/e8d058767dfa
"""

import matplotlib.pyplot as plt
import skimage
from PIL import Image, ImageFilter
from skimage import data, io
from skimage.viewer import ImageViewer

# 以文件的方式读取图片
fp = 'images/dancing.jpg'
# 返回一个 Image 对象
img = Image.open(fp)
print(img)
px = img.load()
print(px)
# 访问图片的属性
print(img.format, img.size, img.mode)  # , img.info)
# 显示图像
# img.show()
# 保存图像
img.save('images/dc.jpg')
# 剪切
box = (100, 100, 300, 300)
region = img.crop(box)
# region.show()
# resize 和 rotate
out = img.resize((128, 128))
# out.show()
out = img.rotate(45)
# out.show()
# 使用滤镜
out = img.filter(ImageFilter.DETAIL)
# out.show()
# 浮点运算
# multiply each pixel by 1.2
out = img.point(lambda i: i * 1.2)
# out.show()
# 处理不同波段
# split the image into individual bands
source = img.split()
R, G, B = 0, 1, 2

# select regions where red is less than 100
mask = source[R].point(lambda i: i < 100 and 255)

# process the green band
out = source[G].point(lambda i: i * 0.7)

# paste the processed band back, but only where red was < 100
source[G].paste(out, None, mask)

# build a new multiband image
im = Image.merge(img.mode, source)
# im.show()

# Python Imaging Library 使用笛卡尔坐标系, 使用 (0,0) 表示左上角. 值得注意的是, 坐标点表示的是一个像素的左上角, 而表示像素的中央则是 (0.5,0.5)


# skimage 部分
camera = data.camera()
print(camera, camera.shape, type(camera))

# 打开本地文件
dancing = io.imread(fp)
print(dancing, dancing.shape, type(dancing))
# Images manipulated by scikit-image are simply NumPy arrays
print(dancing.min(), dancing.max(), dancing.mean())
# plt.imshow(dancing)
# plt.show()
viewer = ImageViewer(dancing)
viewer.show()
