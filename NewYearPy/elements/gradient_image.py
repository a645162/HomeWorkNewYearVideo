import math

import numpy as np
import cv2


def generate_gradient_color(start_color: (int, int, int), end_color: (int, int, int), length: int):
    gradient_color = np.zeros((length, 3), dtype=np.uint8)

    # 计算每个像素的颜色值
    for i in range(length):
        # 计算当前行的颜色插值
        r = int((1 - i / float(length - 1)) * start_color[0] + (i / float(length - 1)) * end_color[0])
        g = int((1 - i / float(length - 1)) * start_color[1] + (i / float(length - 1)) * end_color[1])
        b = int((1 - i / float(length - 1)) * start_color[2] + (i / float(length - 1)) * end_color[2])

        gradient_color[i, :] = (b, g, r)

    return gradient_color


def generate_gradient_image(
        start_color: (int, int, int),
        end_color: (int, int, int),
        width: int = 512,
        height: int = 512,
        center: (int, int) = None,
):
    color_count = 256
    gradient_color = generate_gradient_color(start_color, end_color, color_count)

    gradient_image = np.zeros((height, width, 3), dtype=np.uint8)

    if center is None:
        center_x, center_y = width // 2, height // 2
    else:
        center_x, center_y = center

    r1 = math.sqrt((center_x - 0) ** 2 + (center_y - 0) ** 2)
    r2 = math.sqrt((center_x - width) ** 2 + (center_y - 0) ** 2)
    r3 = math.sqrt((center_x - 0) ** 2 + (center_y - height) ** 2)
    r4 = math.sqrt((center_x - width) ** 2 + (center_y - height) ** 2)
    max_r = max(r1, r2, r3, r4)

    for y in range(height):
        for x in range(width):
            r = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            current_color = gradient_color[int(r / max_r * (color_count - 1)), :]
            gradient_image[y, x] = current_color

    return gradient_image


if __name__ == '__main__':
    gradient_image = (
        generate_gradient_image(
            (100, 0, 0), (200, 200, 0),
            800, 600,
            (200, 100)
        )
    )
    # 显示渐变图像
    cv2.imshow('Color Gradient', gradient_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
