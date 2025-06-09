import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm 

# 定义缺陷计算函数
def color_defect(img):
    B, G, R = cv2.split(img)
    return np.sqrt(np.var(R) + np.var(G) + np.var(B))

def haze_defect(img, patch_size=15):
    dark_channel = cv2.min(cv2.min(img[:,:,0], img[:,:,1]), img[:,:,2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    return 1 - np.mean(dark_channel) / 255

def contrast_defect(img_gray, patch_size=32):
    h, w = img_gray.shape
    std_list = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = img_gray[i:i+patch_size, j:j+patch_size]
            std_list.append(np.std(patch))
    mean_std = np.mean(std_list) 
    return 1 / (mean_std + 1e-6)

def illumination_defect(img_gray):
    mean_lum = np.mean(img_gray)
    std_lum = np.std(img_gray)
    return abs(128 - mean_lum) / 128 + std_lum / 128

# 加载数据集中的图像计算缺陷程度
# image_files = glob.glob('D:/Lab/UWR/datasets/LSUI/input/*.[jp][pn]g')  # 
# 定义多个文件夹路径
folders = [
    'D:/Lab/UWR/datasets/LSUI/input/', 
    'D:/Lab/UWR/datasets/OceanDark2_0', 
    'D:/Lab/UWR/datasets/UIEB_Dataset', 
    'D:/Lab/UWR/datasets/underwater_imagenet/trainA', 
    # 'D:/Lab/UWR/datasets/underwater_imagenet/trainB',
]

# 使用列表推导式获取所有文件
image_files = []
for folder in folders:
    image_files.extend(glob.glob(f'{folder}/*.[jp][pn]g'))

color_scores, haze_scores, contrast_scores, illum_scores = [], [], [], []

for file in tqdm(image_files, desc = "Processing images"):
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color_scores.append(color_defect(img))
    haze_scores.append(haze_defect(img))
    contrast_scores.append(contrast_defect(img_gray))
    illum_scores.append(illumination_defect(img_gray))

# 计算均值和标准差
color_mean, color_std = np.mean(color_scores), np.std(color_scores)
haze_mean, haze_std = np.mean(haze_scores), np.std(haze_scores)
contrast_mean, contrast_std = np.mean(contrast_scores), np.std(contrast_scores)
illum_mean, illum_std = np.mean(illum_scores), np.std(illum_scores)

# Z-score标准化并截断负值
# color_norm = [max(0, (x - color_mean)/color_std) for x in color_scores]
# haze_norm = [max(0, (x - haze_mean)/haze_std) for x in haze_scores]
# contrast_norm = [max(0, (x - contrast_mean)/contrast_std) for x in contrast_scores]
# illum_norm = [max(0, (x - illum_mean)/illum_std) for x in illum_scores]

color_norm = [ (x - color_mean)/color_std for x in color_scores]
haze_norm = [(x - haze_mean)/haze_std for x in haze_scores]
contrast_norm = [(x - contrast_mean)/contrast_std for x in contrast_scores]
illum_norm = [(x - illum_mean)/illum_std for x in illum_scores]

# 展示标准化后的结果
for i, file in enumerate(image_files):
    print(f"{file}: Color={color_norm[i]:.2f}, Haze={haze_norm[i]:.2f}, Contrast={contrast_norm[i]:.2f}, Illumination={illum_norm[i]:.2f}")


print(f"Color - Mean: {color_mean:.4f}, Std: {color_std:.4f}")
print(f"Haze - Mean: {haze_mean:.4f}, Std: {haze_std:.4f}")
print(f"Contrast - Mean: {contrast_mean:.4f}, Std: {contrast_std:.4f}")
print(f"Illumination - Mean: {illum_mean:.4f}, Std: {illum_std:.4f}")

# 绘制缺陷分布直方图与拟合的正态分布曲线
def plot_distribution(scores, mean, std, title):
    plt.figure()
    count, bins, ignored = plt.hist(scores, bins=20, density=True, alpha=0.6, color='g')

    # 正态分布曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

# 调用函数绘制
plot_distribution(color_scores, color_mean, color_std, 'Color Defect Distribution')
plot_distribution(haze_scores, haze_mean, haze_std, 'Haze Defect Distribution')
plot_distribution(contrast_scores, contrast_mean, contrast_std, 'Contrast Defect Distribution')
plot_distribution(illum_scores, illum_mean, illum_std, 'Illumination Defect Distribution')

# plot_distribution(color_norm, color_mean, color_std, 'Color Defect Distribution')
# plot_distribution(haze_norm, haze_mean, haze_std, 'Haze Defect Distribution')
# plot_distribution(contrast_norm, contrast_mean, contrast_std, 'Contrast Defect Distribution')
# plot_distribution(illum_norm, illum_mean, illum_std, 'Illumination Defect Distribution')