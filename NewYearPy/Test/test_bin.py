import cv2

img_line = cv2.imread("img2_line_4channel.png")
print(img_line.shape)

ret, thresh = cv2.threshold(img_line[:, :], 40, 255, cv2.THRESH_BINARY)

img_bin = img_line.copy()
for i in range(img_line.shape[0]):
    for j in range(img_line.shape[1]):
        value = img_line[i, j, 0] + img_line[i, j, 1] + img_line[i, j, 2]
        value /= 3
        if value > 20:
            img_bin[i, j, 0] = 255
            img_bin[i, j, 1] = 255
            img_bin[i, j, 2] = 255
        else:
            img_bin[i, j, 0] = 0
            img_bin[i, j, 1] = 0
            img_bin[i, j, 2] = 0

cv2.imshow("thresh", thresh)
cv2.imshow("my bin", img_bin)
cv2.waitKey(0)
