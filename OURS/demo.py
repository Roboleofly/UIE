import os
import cv2
import numpy as np


# 定义暗通道先验去雾算法
def dehaze(image, omega=0.95, t_min=0.1, window_size=15):
    # 归一化
    image = image.astype('float64') / 255
    # 求暗通道
    min_channel = np.amin(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    # 估计大气光
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    search_idx = (-flat_dark).argsort()[:int(flat_dark.size * 0.001)]
    A = np.mean(flat_image[search_idx], axis=0)
    # 计算透射率
    transmission = 1 - omega * (dark_channel / A.max())
    # 限制透射率
    transmission = np.clip(transmission, t_min, 1)
    # 复原图像
    transmission = cv2.resize(transmission, (image.shape[1], image.shape[0]))
    restored = np.empty_like(image)
    for i in range(3):
        restored[:, :, i] = (image[:, :, i] - A[i]) / transmission + A[i]
    restored = np.clip(restored, 0, 1)
    restored = (restored * 255).astype('uint8')
    return restored


# 颜色校正函数（白平衡）
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


# 定义源文件夹和目标文件夹路径
source_folder = 'Images/'  # 替换为您的源文件夹路径
target_folder = 'tmp_2/'  # 替换为您的目标文件夹路径

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有图片文件
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 构建图片的完整路径
        img_path = os.path.join(source_folder, filename)

        # 使用 OpenCV 读取图片
        img = cv2.imread(img_path)

        # 检查图片是否成功读取
        if img is None:
            print(f"无法读取图片：{img_path}")
            continue

        # 进行去雾处理
        dehazed_img = dehaze(img)

        # 进行颜色校正
        corrected_img = white_balance(dehazed_img)

        # 将原图和增强后的图片进行拼接（水平拼接）
        combined_img = cv2.hconcat([img, corrected_img])

        # 保存拼接后的图片到目标文件夹
        save_path = os.path.join(target_folder, filename)
        cv2.imwrite(save_path, combined_img)

        print(f"已处理并保存图片：{filename}")

print("所有图片处理完成并已保存到目标文件夹。")
