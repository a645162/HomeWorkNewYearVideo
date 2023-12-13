import cv2
import numpy as np


def rotation(image: np.ndarray, angle: float):
    """
    旋转图像
    :param image: 图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h))


# 创建透明背景图像
width, height = 220, 220
background = np.zeros((height, width, 4), dtype=np.uint8)

# 设置文字内容和样式
text = "Haomin Kong"

font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
# White
font_color = (255, 255, 255)
# Black
# font_color = (0, 0, 0)
thickness = 2

# 获取文字的大小和基线
(text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

# 计算文字的位置
x = int((width - text_width) / 2)
y = int((height + text_height) / 2)

# 在透明背景上绘制文字
cv2.putText(background, text, (x, y), font_face, font_scale, font_color, thickness, cv2.LINE_AA)

background = rotation(background, 45)

# 显示图像
cv2.imshow('Text on Transparent Background', background)
cv2.waitKey(0)
cv2.destroyAllWindows()
