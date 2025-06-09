import os
import cv2
import numpy as np
import pandas as pd
from math import log2

# 计算信息熵
def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy = -np.sum([p * log2(p) for p in hist_norm if p > 0])
    return entropy

# 计算梯度均值
def calculate_gradient_mean(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return np.mean(magnitude)

# 主函数：遍历文件夹
def process_images(folder_path, output_csv="image_metrics.csv"):
    results = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_exts):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue  # 跳过无法读取的图像
            entropy = calculate_entropy(image)
            gradient_mean = calculate_gradient_mean(image)
            results.append({
                'filename': filename,
                'entropy': entropy,
                'gradient_mean': gradient_mean
            })

    # 写入 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"处理完成，结果保存在 {output_csv}")

# 示例调用
process_images("../data/expri/4I")
