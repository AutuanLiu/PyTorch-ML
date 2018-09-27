"""
Email: autuanliu@163.com
Date: 2018/9/27
Ref: https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, utils

img_name = 'images/dancing.jpg'
image = Image.open(img_name)  # 读出的内容为 PIL 格式
image = transforms.ToTensor()(image)  # 转换为 tensor
print(image, image.shape)


def imgshow(x):
    # x: (M, N, 3): an image with RGB values (float or uint8)
    plt.imshow(np.transpose(x.numpy(), (1, 2, 0)), interpolation='nearest')


# 简单地展示图片
plt.figure(1)
imgshow(image)

# grid 展示图片
img_list = [image, image, image.clone()]
len1 = len(img_list)
out = utils.make_grid(img_list, nrow=len1, padding=5)
plt.figure(2)
imgshow(out)

out = utils.make_grid(img_list, nrow=len1, padding=15, normalize=True)
plt.figure(3)
imgshow(out)

out = utils.make_grid(img_list, nrow=len1, padding=5,
                      normalize=True, scale_each=True)
plt.figure(4)
imgshow(out)

plt.show()

# 保存图片
utils.save_image(image, 'images/img1.png')  # 传入Tensor
utils.save_image(out, 'images/img2.png')  # 传入list
