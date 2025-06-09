import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
from skimage.filters import sobel
from scipy.stats import entropy
import os

# 计算图像块的信息熵
def block_entropy(block):
    hist, _ = np.histogram(block, bins=256, range=(0, 256), density=True)
    return entropy(hist, base=2)

# 计算图像块的平均梯度
def block_gradient(block):
    gradient = sobel(block)
    return np.mean(gradient)

# 图像分块处理函数
def compute_entropy_gradient_map(img, block_size=(32, 32)):
    h, w = img.shape
    bh, bw = block_size
    blocks = view_as_blocks(img, block_size)
    entropy_map = np.zeros((blocks.shape[0], blocks.shape[1]))
    gradient_map = np.zeros_like(entropy_map)

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i, j]
            entropy_map[i, j] = block_entropy(block)
            gradient_map[i, j] = block_gradient(block)

    return entropy_map, gradient_map

# 可视化函数，绘制多个图像的结果
def visualize_multiple_maps(entropy_maps, gradient_maps, titles):
    num_images = len(entropy_maps)
    fig, axs = plt.subplots(2, num_images, figsize=(4 * num_images, 10))

    vmin_entropy = min(map(np.min, entropy_maps))
    vmax_entropy = max(map(np.max, entropy_maps))
    vmin_grad = min(map(np.min, gradient_maps))
    vmax_grad = max(map(np.max, gradient_maps))

    for i in range(num_images):
        im1 = axs[0, i].imshow(entropy_maps[i], cmap='hot', vmin=vmin_entropy, vmax=vmax_entropy)
        axs[0, i].set_title(f'{titles[i]} - Entropy Map')
        plt.colorbar(im1, ax=axs[0, i])

        im2 = axs[1, i].imshow(gradient_maps[i], cmap='cool', vmin=vmin_grad, vmax=vmax_grad)
        axs[1, i].set_title(f'{titles[i]} - Gradient Map')
        plt.colorbar(im2, ax=axs[1, i])

    plt.tight_layout()
    plt.show()

# 主流程
def main():
    image_paths = [
        '../data/expri/4I/2.png',
        '../data/expri/4I/2_wb1.jpg',
        '../data/expri/4I/2_dehaze2.jpg',
        '../data/expri/4I/2_clahe3.jpg',
        '../data/expri/4I/2_ssr4.jpg',
        # '../data/expri/4I/2_res.jpg'
    ]

    entropy_maps = []
    gradient_maps = []
    titles = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"图像加载失败：{path}")
            continue

        h, w = img.shape
        img_cropped = img[:h - h % 32, :w - w % 32]  # 保证能整除

        entropy_map, gradient_map = compute_entropy_gradient_map(img_cropped, (32, 32))
        entropy_maps.append(entropy_map)
        gradient_maps.append(gradient_map)
        titles.append(os.path.basename(path))

    if entropy_maps:
        visualize_multiple_maps(entropy_maps, gradient_maps, titles)
    else:
        print("没有可用的图像进行处理。")

if __name__ == '__main__':
    main()

# 主流程
