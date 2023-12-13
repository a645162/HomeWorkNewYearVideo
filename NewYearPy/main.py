import cv2
from utils import cv_alpha

from elements import gradient_image, shmtu_logo

# 读取底图
background = gradient_image.generate_gradient_image(
    (0, 0, 0), (255, 136, 0),
    800, 600,
    (200, 100)
)

background = cv_alpha.cv_add_alpha_chanel(background)

# 读取要合成的图像
overlay = shmtu_logo.output_shmtu_logo_image(
    shmtu_logo.image_ori,
    (200, 200)
)

overlay_height, overlay_width = overlay.shape[:2]

# 指定合成图像的位置
x = 100
y = 200

# 提取 overlay 图像的 alpha 通道
overlay_alpha = overlay[:, :, 3] / 255.0

# 根据 alpha 通道对图像进行融合
for c in range(0, 3):
    background[y:y+overlay_height, x:x+overlay_width, c] = \
        overlay_alpha * overlay[:, :, c] + \
        (1 - overlay_alpha) * background[y:y+overlay_height, x:x+overlay_width, c]

# 显示合成后的图像
cv2.imshow('Composite Image', background)
cv2.waitKey(0)
cv2.destroyAllWindows()
