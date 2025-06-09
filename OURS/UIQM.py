import numpy as np
import cv2 as cv
from skimage import io, color, filters
import math
# np.seterr(divide='ignore',invalid='ignore')

def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    #1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int_(al1 * len(rgl))
    T2 = np.int_(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    #2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel=np.round(Rsobel).astype(np.uint8)
    Gsobel=np.round(Gsobel).astype(np.uint8)
    Bsobel=np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    #3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm

    return uiqm


def eme(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]

            blockmin = np.float_(np.min(block))
            blockmax = np.float_(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin += 1
            if blockmax == 0: blockmax += 1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]
            blockmin = np.float_(np.min(block))
            blockmax = np.float_(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            if abs(bottom) < 1e-6:
                bottom = 1e-5

            m = top / bottom
            if m == 0.:
                s += 0
            else:
                s += (m) * np.log(m)

    return plipmult(w, s)



if __name__=="__main__":
    img_0 = cv.imread("../data/expri/11.jpg")
    img_1 = cv.imread("../data/expri/12.jpg")
    #img_2 = cv.imread("compare_imgs/RGHS/R_20.jpg")
    #img_3 = cv.imread("compare_imgs/Fungan/F_20.png")
    #img_4 = cv.imread("compare_imgs/Color/C_20.jpg")
    #img_5 = cv.imread("compare_imgs/Autic/A_20.jpg")
    #img_6 = cv.imread("compare_imgs/Ours/S_20.jpg")

    d_0 = nmetrics(img_0)
    d_1 = nmetrics(img_1)
    #d_2 = nmetrics(img_2)
    #d_3 = nmetrics(img_3)
    #d_4 = nmetrics(img_4)
    #d_5 = nmetrics(img_5)
    #d_6 = nmetrics(img_6)

    print(d_0)
    print(d_1)
    #print(d_2)
    #print(d_3)
    #print(d_4)
    #print(d_5)
    #print(d_6)
