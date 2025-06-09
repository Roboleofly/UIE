# coding = utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images
# img1 = cv2.imread('../data/expri/pair/1.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('../data/expri/pair/2.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('../data/expri/pair/1.jpg')
img2 = cv2.imread('../data/expri/pair/2.jpg')


def draw_custom_matches(img1, kp1, img2, kp2, matches, line_thickness=2):
    # 确保输入图像为彩色
    if len(img1.shape) < 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) < 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # 创建一个足够大的画布以放置两幅图像
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # 将图像放置在画布上
    out_img[:rows1, :cols1] = img1
    out_img[:rows2, cols1:cols1 + cols2] = img2

    # 遍历匹配，绘制线和点
    for mat in matches:
        # 获取匹配点的坐标
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # 绘制关键点
        cv2.circle(out_img, (int(x1), int(y1)), 4, (255, 0, 0), 1)  # Red color for keypoints
        cv2.circle(out_img, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)  # Red color for keypoints

        # 绘制线
        cv2.line(out_img, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 255, 0),
                 line_thickness)  # Green color for lines

    return out_img


# 示例使用
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

print(matches)

# Use rate to filter
good_matches = []
for m, n in matches:
    if m.distance < 0.60 * n.distance:
        good_matches.append(m)

matches = sorted(matches, key=lambda x: x.distance)

result_image = draw_custom_matches(img1, kp1, img2, kp2, matches, line_thickness=3)
cv2.imshow('Custom Match Drawing', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()








