import cv2
import numpy as np
import matplotlib.pyplot as plt


# 使用plt展示OpenCV的彩色图
def cv_show(image, title=""):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 转为uint8
    image_rgb = image_rgb.astype(np.uint8)

    plt.imshow(image_rgb)
    plt.axis('off')
    if len(title) > 0:
        plt.title(title)
    plt.show()


# 使用plt展示OpenCV的灰度图
def cv_show_gray(image, title=""):
    # 转为uint8
    image = image.astype(np.uint8)

    plt.imshow(image, cmap='gray')
    plt.axis('off')
    if len(title) > 0:
        plt.title(title)
    plt.show()
