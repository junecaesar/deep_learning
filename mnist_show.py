# coding: utf-8
import sys, os
#sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print(len(x_train), len(t_train), len(x_test), len(t_test))

for counter in range(50):
    print("Please input the line number you wanna show:")
    line_number=int(input())
    if(line_number==139):
        break
    if (line_number<1 or line_number>10000):
        print("Not in range 1-10000")
        continue

#line_number=10
    img = x_train[line_number-1]
    label = t_train[line_number-1]
    print("现在显示第%s行数字，标签为%s"%(line_number,label))  # show the number

#    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
    print("点阵格式为",img.shape)  # (28, 28)

    img_show(img) #show the piture of the number
