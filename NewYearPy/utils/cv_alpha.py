import cv2
import numpy as np


def cv_add_alpha_chanel(image: np.ndarray):
    """
    为图像添加 alpha 通道
    :param image: 图像
    :return: 添加了 alpha 通道的图像
    """

    alpha = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255

    return cv2.merge((image, alpha))
