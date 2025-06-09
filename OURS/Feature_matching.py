# coding = utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Load two images
img1 = cv2.imread('../data/expri/pair/3.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../data/expri/pair/4.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('../data/expri/pair/3.jpg')
img4 = cv2.imread('../data/expri/pair/4.jpg')

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Use SIFT detector to find the features and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Create BF matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k = 2)

# Use rate to filter
good_matches = []
for m, n in matches:
    if m.distance < 0.60 * n.distance:
        good_matches.append(m)

print(f'Number of good matches: {len(good_matches)}')

# Draw the result
img_3 = draw_custom_matches(img3, keypoints1, img4, keypoints2, good_matches, line_thickness=3)

# # 手动绘制连线
# for match in good_matches:
#     # 随机生成颜色
#     color = np.random.randint(0, 255, 3).tolist()
#
#     pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype(int))
#     pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
#     cv2.line(img3, pt1, pt2, color, thickness= 5 )  # 可以调整thickness来改变连线粗细



cv2.imwrite('temp_2.jpg', img_3)

# Display the result
plt.imshow(img_3)
plt.show()








