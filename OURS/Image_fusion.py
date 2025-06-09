import cv2 as cv
import numpy as np
from White_balance import White_balance
from Dehaze import dehaze
from CLAHE import CLAHE
from SSR import SSR
from SimplestColorBalance import simplest_cb
from FeatureWeight import LaplacianContrast_2, LocalContrast_2, Saliency_2
from ImageDecompose import fuseTwoImages, fuseFourImages, fuseThreeImages
from UDCP import UDCP
import os

def calWeight(img, L):
    L = np.float32(np.divide(L, (255.0)))
    WL = np.float32(LaplacianContrast_2(L)) # Check this line
    WC = np.float32(LocalContrast_2(img))
    WS = np.float32(Saliency_2(img))
    # WE = np.float32(Exposedness(L))

    weight = WL.copy()
    weight = np.add(weight, WC)
    weight = np.add(weight, WS)
    # weight = np.add(weight, WE)
    return weight

'''
def Weighted_factors(w_list):
    w_11 = 0.8 * w_list[0]
    w_22 = 3.0 * w_list[1] - 0.1
    w_33 = 2.0 * w_list[2]
    w_44 = w_list[3] - 0.3

    if w_22 < 0: w_22 = 0.0
    if w_33 < 0: w_33 = 0.0
    if w_44 < 0: w_44 = 0.0


    raw_w_list = [w_11, w_22, w_33, w_44]

    print("The raw weighted params:\n")
    print(raw_w_list)

    normalized_w_list = raw_w_list / np.sum(raw_w_list)

    print("The normalized weighted params:\n")
    print(normalized_w_list)

    return normalized_w_list
'''

'''
def Enhance_img(img_0):
    #img_1 = img_0.copy()

    img1 = simplest_cb(img_0, 5)
    img_1, w_1 = White_balance(img_0)

    img_2 = img1.copy()
    img_3 = img1.copy()
    img_4 = img1.copy()

    temp_img2, w_2 = DeHaze(img_2)
    img2 = UDCP(img_2)
    img3, w_3 = CLAHE(img_3)
    img4, w_4 = SSR(img_4, 0, 3.0)

    # w_list = [w_1, w_2, w_3, w_4]

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

    w1 = calWeight(img1, L1) #* new_w_list[0]
    w2 = calWeight(img2, L2) #* new_w_list[1]
    w3 = calWeight(img3, L3) #* new_w_list[2]
    w4 = calWeight(img4, L4) #* new_w_list[3]


    sumW = w1 + w2 + w3 + w4
    w1 = cv.divide(w1, sumW)
    w2 = cv.divide(w2, sumW)
    w3 = cv.divide(w3, sumW)
    w4 = cv.divide(w4, sumW)

        # max_1 = np.max(w1) / 1e3
        # max_2 = np.max(w2) / 1e3
        # max_3 = np.max(w3) / 1e3
        # max_4 = np.max(w4) / 1e3
        # 
        # w1 = w1 / max_1
        # w2 = w2 / max_2
        # w3 = w3 / max_3
        # w4 = w4 / max_4
        # 
        # cv.imshow('w1', w1)
        # cv.imshow('w2', w2)
        # cv.imshow('w3', w3)
        # cv.imshow('w4', w4)
        
    level = 5
    #cv.imshow("Img_2", img1)
    #cv.imshow("Img_2", img2)
    #cv.imshow("img_3", img3)
    #cv.imshow("img_4", img4)

    cv.imwrite("raw_1.jpg", img1)
    cv.imwrite("raw_2.jpg", img2)
    cv.imwrite("raw_3.jpg", img3)
    cv.imwrite("raw_4.jpg", img4)

    res_img = fuseFourImages(w1, img1, w2, img2, w3, img3, w4, img4, level)

    res_img = np.uint8(res_img)
    # cv.imshow("Original", img_0)
    # cv.imshow("Fusion", res_img )

    return res_img
'''

def Enhance_img_path(img_0):
    #img_1 = img_0.copy()

    img1 = simplest_cb(img_0, 5)
    img_1, w_1 = White_balance(img_0)

    img_2 = img1.copy()
    img_3 = img1.copy()
    img_4 = img1.copy()

    img2 = dehaze(img_2)
    # img2 = UDCP(img_2)
    img3, w_3 = CLAHE(img_3)
    img4, w_4 = SSR(img_4, 0, 5.0)

    save_path = 'demo/new'
    cv.imwrite(save_path + '_wb1.jpg', img1)
    cv.imwrite(save_path + '_dehaze2.jpg', img2)
    cv.imwrite(save_path + '_clahe3.jpg', img3)
    cv.imwrite(save_path + '_ssr4.jpg', img4)


    # w_list = [w_1, w_2, w_3, w_4]
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




    detect_factor = [0.3, 0.3, 0.3, 0.1]


    w1 = calWeight(img1, L1) * detect_factor[0]
    w2 = calWeight(img2, L2) * detect_factor[1]
    w3 = calWeight(img3, L3) * detect_factor[2]
    w4 = calWeight(img4, L4) * detect_factor[3]


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

if __name__ ==   '__main__':

    # img_0 = cv.imread("../data/expri/1.png")
    # save_path = "../data/expri/1"

    img_0 = cv.imread('demo/8.jpg')

    Enhance_img_path(img_0)




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







