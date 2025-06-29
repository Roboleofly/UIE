from .FeatureWeight import LaplacianContrast, LocalContrast, Saliency, Exposedness
from .ImageDecompose import fuseTwoImages
from .SimplestColorBalance import simplest_cb
import numpy as np
import cv2
import time

def enhance(image, level):
    img1 = simplest_cb(image, 5)
    img1 = np.uint8(img1)
    LabIm1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    L1 = cv2.extractChannel(LabIm1, 0)
    # Apply CLAHE
    result = applyCLAHE(LabIm1, L1)
    img2 = result[0]
    L2 = result[1]
    w1 = calWeight(img1, L1)
    w2 = calWeight(img2, L2)
    sumW = cv2.add(w1, w2)
    w1 = cv2.divide(w1, sumW)
    w2 = cv2.divide(w2, sumW)
    return fuseTwoImages(w1, img1, w2, img2, level)  


def applyCLAHE(img, L):
    clahe = cv2.createCLAHE(clipLimit=2.0)
    L2 = clahe.apply(L)
    lab = cv2.split(img)
    LabIm2 = cv2.merge((L2, lab[1], lab[2]))
    img2 = cv2.cvtColor(LabIm2, cv2.COLOR_Lab2BGR)
    result = []
    result.append(img2)
    result.append(L2)
    return result


def calWeight(img, L):
    L = np.float32(np.divide(L, (255.0)))
    WL = np.float32(LaplacianContrast(L)) # Check this line
    WC = np.float32(LocalContrast(L))
    WS = np.float32(Saliency(img))
    WE = np.float32(Exposedness(L))
    weight = WL.copy()
    weight = np.add(weight, WC)
    weight = np.add(weight, WS)
    weight = np.add(weight, WE)
    return weight


if __name__ ==   '__main__':
    path = "../data/demo/4.png"
    level = 5
    start = time.time()
    image = cv2.imread(path)
    fusion = enhance(image, 5)
    print(time.time()-start)
    fusion= np.uint8(fusion)
    cv2.imshow("Original", image)
    #cv2.imwrite("../data/demo/fusion_1_2.jpg",fusion)
    cv2.imshow("Fusion", fusion)
    cv2.waitKey(0)
