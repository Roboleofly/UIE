#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from PIL import Image
import numpy as np
from metrics import calculate_uciqe, calculate_uiqm


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


uibe_imge_list = ['2629.png', '789_img_.png', '9_img_.png', '534_img_.png', '614_img_.png', '155_img_.png', '696_img_.png', '12324.png', '332_img_.png', '682_img_.png', '793_img_.png', '880_img_.png', '767_img_.png', '451_img_.png', '15426.png', '287_img_.png', '917_img_.png', '1573.png', '12348.png', '320_img_.png', '555_img_.png', '400_img_.png', '765_img_.png', '53_img_.png', '366_img_.png', '834_img_.png', '1660.png', '531_img_.png', '225_img_.png', '611_img_.png', '12290.png', '111_img_.png', '913_img_.png', '491_img_.png', '715_img_.png', '632_img_.png', '923_img_.png', '869_img_.png', '580_img_.png', '661_img_.png', '709_img_.png', '625_img_.png', '160_img_.png', '221_img_.png', '615_img_.png', '385_img_.png', '140_img_.png', '100_img_.png', '306_img_.png', '763_img_.png', '798_img_.png', '918_img_.png', '290_img_.png', '294_img_.png', '258_img_.png', '893_img_.png', '841.png', '784_img_.png', '315_img_.png']
oceandark_image_list = ['150.jpg', '69.jpg', '36.jpg', '67.jpg', '159.jpg', '88.jpg', '155.jpg', '158.jpg', '160.jpg', '157.jpg', '37.jpg', '68.jpg', '35.jpg', '89.jpg', '3.jpg']

uibe_pre = '/media/users/leo/workspace/UIE/expri/UIEB_Dataset'
oceandark_pre = '/media/users/leo/workspace/UIE/expri/OceanDark2_0'


algorithms = ['COLOR', 'FGAN', 'FUSION2', 'OURS', 'RGHS', 'UDCP', 'UGAN', 'ULAP']

# 
color_metrics = []
fgan_metics = []
fusion2_metrics = []
rghs_metrics = []
udcp_metrics = []
ugan_metrics = []
ulap_metrics = []
ours_metrics = []

def cal_raw_metrics():
    entropy_metrics = []
    uiqm_metrics = []
    uciqe_metrics = []

    data_oceandark_pre = '/media/users/leo/workspace/UIE/datasets/OceanDark2_0'
    data_uibe_pre = '/media/users/leo/workspace/UIE/datasets/UIEB_Dataset'

    for image_name in oceandark_image_list:
        image_path = os.path.join(data_oceandark_pre, image_name)
        # print(image_path)
        if os.path.exists(image_path):
            entropy = calculate_entropy(image_path)
            entropy_metrics.append(entropy)
    for image_name in uibe_imge_list:
        image_path = os.path.join(data_uibe_pre, image_name)
        # print(image_path)
        if os.path.exists(image_path):
            entropy = calculate_entropy(image_path)
            entropy_metrics.append(entropy)
    
    for image_name in oceandark_image_list:
        image_path = os.path.join(data_oceandark_pre, image_name)
        # print(image_path)
        if os.path.exists(image_path):
            entropy = calculate_uiqm(image_path)
            uiqm_metrics.append(entropy)
    for image_name in uibe_imge_list:
        image_path = os.path.join(data_uibe_pre, image_name)
        # print(image_path)
        if os.path.exists(image_path):
            entropy = calculate_uiqm(image_path)
            uiqm_metrics.append(entropy)
    
    for image_name in oceandark_image_list:
        image_path = os.path.join(data_oceandark_pre, image_name)
        # print(image_path)
        if os.path.exists(image_path):
            entropy = calculate_uciqe(image_path)
            uciqe_metrics.append(entropy)
    for image_name in uibe_imge_list:  
        image_path = os.path.join(data_uibe_pre, image_name)
        # print(image_path)
        if os.path.exists(image_path):
            entropy = calculate_uciqe(image_path)
            uciqe_metrics.append(entropy)
    
    print('entropy:', np.mean(entropy_metrics))
    print('uiqm:', np.mean(uiqm_metrics))
    print('uciqe:', np.mean(uciqe_metrics))


def cal_average_entropy():
    # 计算平均熵
    for algorithm in algorithms:
        for image_name in oceandark_image_list:
            image_path = os.path.join(oceandark_pre, algorithm, image_name)
            print(image_path)
            if os.path.exists(image_path):
                entropy = calculate_entropy(image_path)
                if algorithm == 'COLOR':
                    color_metrics.append(entropy)
                elif algorithm == 'FGAN':
                    fgan_metics.append(entropy)
                elif algorithm == 'FUSION2':
                    fusion2_metrics.append(entropy)
                elif algorithm == 'OURS':
                    ours_metrics.append(entropy)
                elif algorithm == 'RGHS':
                    rghs_metrics.append(entropy)
                elif algorithm == 'UDCP':
                    udcp_metrics.append(entropy)
                elif algorithm == 'UGAN':
                    ugan_metrics.append(entropy)
                elif algorithm == 'ULAP':
                    ulap_metrics.append(entropy)

        for image_name in uibe_imge_list:
            image_path = os.path.join(uibe_pre, algorithm, image_name)
            print(image_path)
            if os.path.exists(image_path):
                entropy = calculate_entropy(image_path)
                if algorithm == 'COLOR':
                    color_metrics.append(entropy)
                elif algorithm == 'FGAN':
                    fgan_metics.append(entropy)
                elif algorithm == 'FUSION2':
                    fusion2_metrics.append(entropy)
                elif algorithm == 'OURS':
                    ours_metrics.append(entropy)
                elif algorithm == 'RGHS':
                    rghs_metrics.append(entropy)
                elif algorithm == 'UDCP':
                    udcp_metrics.append(entropy)
                elif algorithm == 'UGAN':
                    ugan_metrics.append(entropy)
                elif algorithm == 'ULAP':
                    ulap_metrics.append(entropy)

    print('color:', np.mean(color_metrics))
    print('fgan:', np.mean(fgan_metics))
    print('fusion2:', np.mean(fusion2_metrics))
    print('ours:', np.mean(ours_metrics))
    print('rghs:', np.mean(rghs_metrics))
    print('udcp:', np.mean(udcp_metrics))
    print('ugan:', np.mean(ugan_metrics))
    print('ulap:', np.mean(ulap_metrics))

    # print(color_metrics)


def cal_average_uiqm():
    # 计算平均熵
    for algorithm in algorithms:
        for image_name in oceandark_image_list:
            image_path = os.path.join(oceandark_pre, algorithm, image_name)
            print(image_path)
            if os.path.exists(image_path):
                entropy = calculate_uiqm(image_path)
                if algorithm == 'COLOR':
                    color_metrics.append(entropy)
                elif algorithm == 'FGAN':
                    fgan_metics.append(entropy)
                elif algorithm == 'FUSION2':
                    fusion2_metrics.append(entropy)
                elif algorithm == 'OURS':
                    ours_metrics.append(entropy)
                elif algorithm == 'RGHS':
                    rghs_metrics.append(entropy)
                elif algorithm == 'UDCP':
                    udcp_metrics.append(entropy)
                elif algorithm == 'UGAN':
                    ugan_metrics.append(entropy)
                elif algorithm == 'ULAP':
                    ulap_metrics.append(entropy)

        for image_name in uibe_imge_list:
            image_path = os.path.join(uibe_pre, algorithm, image_name)
            print(image_path)
            if os.path.exists(image_path):
                entropy = calculate_uiqm(image_path)
                if algorithm == 'COLOR':
                    color_metrics.append(entropy)
                elif algorithm == 'FGAN':
                    fgan_metics.append(entropy)
                elif algorithm == 'FUSION2':
                    fusion2_metrics.append(entropy)
                elif algorithm == 'OURS':
                    ours_metrics.append(entropy)
                elif algorithm == 'RGHS':
                    rghs_metrics.append(entropy)
                elif algorithm == 'UDCP':
                    udcp_metrics.append(entropy)
                elif algorithm == 'UGAN':
                    ugan_metrics.append(entropy)
                elif algorithm == 'ULAP':
                    ulap_metrics.append(entropy)

    print('color:', np.mean(color_metrics))
    print('fgan:', np.mean(fgan_metics))
    print('fusion2:', np.mean(fusion2_metrics))
    print('ours:', np.mean(ours_metrics))
    print('rghs:', np.mean(rghs_metrics))
    print('udcp:', np.mean(udcp_metrics))
    print('ugan:', np.mean(ugan_metrics))
    print('ulap:', np.mean(ulap_metrics))

    # print(color_metrics)


def cal_average_uciqe():
    # 计算平均熵
    for algorithm in algorithms:
        for image_name in oceandark_image_list:
            image_path = os.path.join(oceandark_pre, algorithm, image_name)
            print(image_path)
            if os.path.exists(image_path):
                entropy = calculate_uciqe(image_path)
                if algorithm == 'COLOR':
                    color_metrics.append(entropy)
                elif algorithm == 'FGAN':
                    fgan_metics.append(entropy)
                elif algorithm == 'FUSION2':
                    fusion2_metrics.append(entropy)
                elif algorithm == 'OURS':
                    ours_metrics.append(entropy)
                elif algorithm == 'RGHS':
                    rghs_metrics.append(entropy)
                elif algorithm == 'UDCP':
                    udcp_metrics.append(entropy)
                elif algorithm == 'UGAN':
                    ugan_metrics.append(entropy)
                elif algorithm == 'ULAP':
                    ulap_metrics.append(entropy)

        for image_name in uibe_imge_list:
            image_path = os.path.join(uibe_pre, algorithm, image_name)
            print(image_path)
            if os.path.exists(image_path):
                entropy = calculate_uciqe(image_path)
                if algorithm == 'COLOR':
                    color_metrics.append(entropy)
                elif algorithm == 'FGAN':
                    fgan_metics.append(entropy)
                elif algorithm == 'FUSION2':
                    fusion2_metrics.append(entropy)
                elif algorithm == 'OURS':
                    ours_metrics.append(entropy)
                elif algorithm == 'RGHS':
                    rghs_metrics.append(entropy)
                elif algorithm == 'UDCP':
                    udcp_metrics.append(entropy)
                elif algorithm == 'UGAN':
                    ugan_metrics.append(entropy)
                elif algorithm == 'ULAP':
                    ulap_metrics.append(entropy)

    print('color:', np.mean(color_metrics))
    print('fgan:', np.mean(fgan_metics))
    print('fusion2:', np.mean(fusion2_metrics))
    print('ours:', np.mean(ours_metrics))
    print('rghs:', np.mean(rghs_metrics))
    print('udcp:', np.mean(udcp_metrics))
    print('ugan:', np.mean(ugan_metrics))
    print('ulap:', np.mean(ulap_metrics))

    # print(color_metrics)


if __name__ == '__main__':

    # if len(sys.argv) != 3:
    #     print("用法: python compare_entropy.py <dir_1> <dir_2>")
    #     sys.exit(1)

    # cal_average_entropy()
    # cal_average_uiqm()

    # len_uibe = len(uibe_imge_list)
    # len_oceandark = len(oceandark_image_list)
    # total_lem = len_uibe + len_oceandark
    # print("len_uibe: ", len_uibe)
    # print("len_oceandark ", len_oceandark)
    # print("Percentage of uibe: ", len_uibe/total_lem)
    # print("Percentage of oceandark ", len_oceandark/total_lem)

    # cal_average_uciqe()

    cal_raw_metrics()
