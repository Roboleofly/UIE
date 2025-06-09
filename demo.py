#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from PIL import Image
import numpy as np

def calculate_entropy(image_path):
    """
    计算单张灰度图像的香农熵：
      H = -∑ p(i) log2 p(i)
    """
    img = Image.open(image_path).convert('L')
    hist = np.array(img.histogram(), dtype=np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def main(dir1, dir2):
    # 支持的图片后缀
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif')
    # 结果列表
    higher_in_dir2 = []

    # 枚举 dir1 中的图片文件
    for fname in os.listdir(dir1):
        if not fname.lower().endswith(exts):
            continue

        path1 = os.path.join(dir1, fname)
        path2 = os.path.join(dir2, fname)

        # 只处理在 dir2 中也存在同名文件的情况
        if not os.path.isfile(path2):
            continue

        try:
            e1 = calculate_entropy(path1)
            e2 = calculate_entropy(path2)
        except Exception as ex:
            print(f"无法处理 {fname}：{ex}")
            continue

        # 如果 dir_2 的熵更大，则加入列表
        if e2 > e1:
            higher_in_dir2.append(fname)

    # 打印结果
    if higher_in_dir2:
        print(higher_in_dir2)
        # print("以下图片在 dir_2 中的熵高于 dir_1：")
        # for name in higher_in_dir2:
        #     print(f"  {name}")
    else:
        print("没有找到在 dir_2 中熵更高的同名图片。")

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("用法: python compare_entropy.py <dir_1> <dir_2>")
    #     sys.exit(1)

    # dir_1 = '/media/users/leo/workspace/UIE/expri/UIEB/FUSION2'
    # dir_2 = '/media/users/leo/workspace/UIE/expri/UIEB/OURS'
    dir_1 = '/media/users/leo/workspace/UIE/expri/OceanDark2_0/FUSION2'
    dir_2 = '/media/users/leo/workspace/UIE/expri/OceanDark2_0/OURS'

    if not os.path.isdir(dir_1) or not os.path.isdir(dir_2):
        print("错误：请确保两个参数都是已存在的目录。")
        sys.exit(1)

    main(dir_1, dir_2)
