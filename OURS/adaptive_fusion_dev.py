import cv2 as cv
import numpy as np
from .White_balance import White_balance
from .Dehaze import dehaze
from .CLAHE import CLAHE
from .SSRv2 import SSRv2
from .SSR import SSR
from .SimplestColorBalance import simplest_cb
from .FeatureWeight import LaplacianContrast, LocalContrast, Saliency, Exposedness
from .ImageDecompose import fuseTwoImages, fuseFourImages, fuseThreeImages
from .UDCP import UDCP
import os
from tqdm import tqdm 
import glob

# 定义缺陷计算函数
def color_defect(img):
    B, G, R = cv.split(img)
    return np.sqrt(np.var(R) + np.var(G) + np.var(B))

def haze_defect(img):
    # 使用蓝绿通道的暗通道均值
    B, G, _ = cv.split(img)
    min_bg = np.minimum(B, G)
    dark_channel = cv.erode(min_bg, np.ones((15, 15)))  # 局部最小滤波
    return 1 - np.mean(dark_channel) / 255.0

def contrast_defect(img_gray):
    p20 = np.percentile(img_gray, 20)
    p80 = np.percentile(img_gray, 80)
    return 1 / (p80 - p20 + 1e-6)  # 分位差越大对比度越好，取倒数表示缺陷

def illumination_defect(img_gray):
    mean_lum = np.mean(img_gray)
    std_lum = np.std(img_gray)
    return abs(128 - mean_lum) / 128 + std_lum / 128


def calWeight(img, L):
    L = np.float32(np.divide(L, (255.0)))
    WL = np.float32(LaplacianContrast(L)) # Check this line
    WC = np.float32(LocalContrast(img))
    WS = np.float32(Saliency(img))
    WE = np.float32(Exposedness(L))

    weight = WL.copy()
    weight = np.add(weight, WC)
    weight = np.add(weight, WS)
    weight = np.add(weight, WE)

    return weight

def normalize_3sigma(x, mu, sigma):
    lower = mu - 3 * sigma
    upper = mu + 3 * sigma
    norm = (x - lower) / (upper - lower)
    return np.clip(norm, 0, 1)  # 限制在 [0, 1] 区间

def Enhance_img(img_0):
    #img_1 = img_0.copy()

    img1 = simplest_cb(img_0, 5)
    # img1, w_1 = White_balance(img_0)

    img_2 = img1.copy()
    img_3 = img1.copy()
    img_4 = img1.copy()

    img2 = dehaze(img_2)
    # img2 = UDCP(img_2)
    img3, w_3 = CLAHE(img_3)
    # img4, w_4 = SSR(img_4, 1, 5.0)
    img4 = SSRv2(img_4)
    # cv.imshow('t', img4)
    # cv.waitKey(0)

    # save_path = 'demo/new'
    # cv.imwrite(save_path + '_wb1.jpg', img1)
    # cv.imwrite(save_path + '_dehaze2.jpg', img2)
    # cv.imwrite(save_path + '_clahe3.jpg', img3)
    # cv.imwrite(save_path + '_ssr4.jpg', img4)


    # calculate the detect_factor 
    img_0_gray = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY)
    color_scores = color_defect(img_0)
    haze_scores = haze_defect(img_0)
    contrast_scores = contrast_defect(img_0_gray)
    illum_scores = illumination_defect(img_0_gray)

    # record the statistic value 
    color_mean = 76.2560 
    color_std = 22.9359
    haze_mean = 0.7229
    haze_std = 0.1289
    contrast_mean = 0.0175
    contrast_std = 0.0144
    illum_mean = 0.5437
    illum_std = 0.1820

    # 3sigma norm 
    color_norm = normalize_3sigma(color_scores, color_mean, color_std) 
    haze_norm = normalize_3sigma(haze_scores, haze_mean, haze_std)
    contrast_norm = normalize_3sigma(contrast_scores, contrast_mean, contrast_std) 
    illum_norm = normalize_3sigma(illum_scores, illum_mean, illum_std)

    # w_list = [w_1, w_2, w_3, w_4]
    defect_factor  = [color_norm, haze_norm, contrast_norm, illum_norm]

    defect_factor = np.float32(defect_factor)

    print(defect_factor)
    
    # new_w_list = Weighted_factors(w_list)

    img1 = np.uint8(img1)
    LabIm1 = cv.cvtColor(img1, cv.COLOR_BGR2Lab)
    L1 = cv.extractChannel(LabIm1, 0)

    img2 = np.uint8(img2)
    LabIm2 = cv.cvtColor(img2, cv.COLOR_BGR2Lab)
    L2 = cv.extractChannel(LabIm2, 0)

    img3 = np.uint8(img3)
    LabIm3 = cv.cvtColor(img3, cv.COLOR_BGR2Lab)
    L3 = cv.extractChannel(LabIm3, 0)

    img4 = np.uint8(img4)
    LabIm4 = cv.cvtColor(img4, cv.COLOR_BGR2Lab)
    L4 = cv.extractChannel(LabIm4, 0)

    #cv.imshow('L1', L1)
    #cv.imshow('L2', L2)
    #cv.imshow('L3', L3)
    #cv.imshow('L4', L4)


    # detect_factor = [0.3, 0.3, 0.3, 0.1]
    # defect_factor = [0.25, 0.25, 0.25, 0.25]


    w1 = calWeight(img1, L1) * defect_factor[0] 
    w2 = calWeight(img2, L2) * defect_factor[1] 
    w3 = calWeight(img3, L3) * defect_factor[2]
    w4 = calWeight(img4, L4) * defect_factor[3] 


    sumW = w1 + w2 + w3 + w4
    w1 = cv.divide(w1, sumW)
    w2 = cv.divide(w2, sumW)
    w3 = cv.divide(w3, sumW)
    w4 = cv.divide(w4, sumW)

    '''
        max_1 = np.max(w1) / 1e3
        max_2 = np.max(w2) / 1e3
        max_3 = np.max(w3) / 1e3
        max_4 = np.max(w4) / 1e3

        w1 = w1 / max_1
        w2 = w2 / max_2
        w3 = w3 / max_3
        w4 = w4 / max_4

        cv.imshow('w1', w1)
        cv.imshow('w2', w2)
        cv.imshow('w3', w3)
        cv.imshow('w4', w4)
    '''

    level = 5
    #cv.imshow("Img_2", img1)
    #cv.imshow("Img_2", img2)
    #cv.imshow("img_3", img3)
    #cv.imshow("img_4", img4)

    # cv.imwrite("raw_1.jpg", img1)
    # cv.imwrite("raw_2.jpg", img2)
    # cv.imwrite("raw_3.jpg", img3)
    # cv.imwrite("raw_4.jpg", img4)

    res_img = fuseFourImages(w1, img1, w2, img2, w3, img3, w4, img4, level)
    res_img = np.uint8(res_img)
    # cv.imshow("Original", img_0)
    # cv.imshow("Fusion", res_img )

    return res_img


def process_images(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Use glob to find all image files in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.png")) + glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.jpeg"))
    
    print(image_paths)
    
    # Process each image with tqdm progress bar
    for input_path in tqdm(image_paths, desc="Processing Images"):
        filename = os.path.basename(input_path)  # Get the filename
        
        img = cv.imread(input_path)
        
        if img is None:
            continue  # Skip if the image is not readable
        
        # Process the image
        res = Enhance_img(img)
        
        # Combine original and enhanced image for display
        combined = cv.hconcat([img, res])
        # combined_resized = cv.resize(combined, [1280, 640])
        
        # Display the result
        # cv.imshow('Processed Image', combined_resized)
        
        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, f"enhanced_{filename}")
        cv.imwrite(output_path, combined)  # Save only the enhanced image
        


if __name__ ==   '__main__':

    # img_0 = cv.imread("../data/expri/1.png")
    # save_path = "../data/expri/1"


    # img_0 = cv.imread('demo/10.png')

    # res = Enhance_img_path(img_0)

    # cv.imshow('tmp', cv.resize(cv.hconcat([img_0, res]), [1280,640]))

    # cv.waitKey(0)

    img = cv.imread('/media/users/leo/workspace/UIE/datasets/UIDEF_S/Blur_Perception_00003_Img.jpg')
    res = Enhance_img(img)



    # process in folder <----- TODO
    # input_dir = 'Raw'
    # output_dir = 'Res2'
    # process_images(input_dir, output_dir)




    # source_folder = 'Images/'  # 替换为您的源文件夹路径
    # target_folder = 'tmp_3/'  # 替换为您的目标文件夹路径

    # 如果目标文件夹不存在，则创建它
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder)

    # for filename in os.listdir(source_folder):
    #     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    #         # 构建图片的完整路径
    #         img_path = os.path.join(source_folder, filename)

    #         img = cv.imread(img_path)

    #         enhanced_img = Enhance_img_path(img)

    #         # 拼接图片（水平拼接）
    #         combined_img = cv.hconcat([img, enhanced_img])

    #         # 保存图片
    #         # 保存拼接后的图片到目标文件夹
    #         save_path = os.path.join(target_folder, filename)
    #         cv.imwrite(save_path, combined_img)

    #         print("Save " + save_path)

    # print("Done ! ")

    # Change the saturation
    # hsv_res_img = cv.cvtColor(res_img, cv.COLOR_BGR2HSV)
    # saturation_scale = 1.001
    # hsv_res_img[..., 1] = hsv_res_img[..., 1] * saturation_scale
    # res_img_1 = cv.cvtColor(hsv_res_img, cv.COLOR_HSV2BGR)
    
    # cv.imwrite(save_path, res_img)
    # print("Done!!!!")
    # print("UCIQE: ")
    # uciqe_temp = uciqe(1, save_path)
    # print(uciqe_temp)
    # print("UIQM: ")
    # uiqm_temp = nmetrics(res_img)
    # print(uiqm_temp)
    # cv.imshow("Enhanced Image", res_img)
    # cv.waitKey(0)

    '''
    file_path_name = "Raw"
    uciqe_list = []
    uiqm_list = []
    for file_name in os.listdir(file_path_name):
        img_0 = cv.imread(file_path_name+'/'+file_name)
        save_path = "res_temp.jpg"
        res_img = Enhane_img(img_0)

        cv.imwrite(save_path,res_img)
        print("Done!!!!")
        print("UCIQE: ")
        uciqe_temp = uciqe(1, save_path)
        print(uciqe_temp)
        uciqe_list.append(uciqe_temp)
        print("UIQM: ")
        uiqm_temp = nmetrics(res_img)
        print(uiqm_temp)
        uiqm_list.append(uiqm_temp)

        #cv.imshow("Enhanced Image", res_img_1)
        #cv.waitKey(0)

        cv.imwrite("Out/"+file_name, res_img)

    print("UCIQE_LIST:")
    print(uciqe_list)
    print("UIQM_LIST")
    print(uiqm_list)
    f = open("temp.txt", "w")
    f.writelines(str(uciqe_list))
    f.writelines(str(uiqm_list))
    f.close()


shape = img1.shape
size = [shape[0],shape[1]]

w1 = np.ones(size) * w_1
w2 = np.ones(size) * w_2
w3 = np.ones(size) * w_3
w4 = np.ones(size) * w_4
w1 = np.float32(w1)
w2 = np.float32(w2)
w3 = np.float32(w3)
w4 = np.float32(w4)
'''







