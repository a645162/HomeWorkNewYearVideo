import cv2
import numpy as np
import os

from config.size import logo_max_width, logo_max_height

# script_path = os.path.abspath(__file__)
# parent_dir = os.path.dirname(script_path)
# parent_dir = os.path.dirname(parent_dir)
#
# path = os.path.join(parent_dir, "res")
# path = os.path.join(path, "shmtu_logo.png")

path = "res/shmtu_logo.png"

if not os.path.exists(path):
    print(path)
    print("Cannot find SHMTU Logo file.")

# 读取图像
image_ori = cv2.imread(path, cv2.IMREAD_UNCHANGED)
image_ori = cv2.resize(image_ori, (logo_max_width, logo_max_height))

size = image_ori.shape[:2]
print(image_ori.shape)
height, width = size
total_pixel = width * height


def output_shmtu_logo_image(image: np.ndarray, dsize: (int, int)):
    # resize
    new_image = image.copy()
    new_image = cv2.resize(new_image, dsize)

    return new_image


if __name__ == '__main__':
    new_pic = output_shmtu_logo_image(image_ori, (1000, 1000))

    cv2.imshow("new_pic", new_pic)
    cv2.waitKey(0)
