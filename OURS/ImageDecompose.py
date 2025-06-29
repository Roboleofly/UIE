import cv2
import numpy

def filterMask(img):
    h = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
    mask = numpy.zeros((len(h), len(h)), img[0][1].dtype)
    for i in range(len(h)):
        for j in range(len(h)):
            mask[i][j] = h[i] * h[j]
    return mask


def buildGaussianPyramid(img, level):
    gaussPyr =[]
    mask = filterMask(img)
    tmp = cv2.filter2D(img, -1, mask)
    gaussPyr.append(tmp.copy())
    tmpImg = img.copy()
    for i in range(1,level):
        cv2.resize(tmpImg, (0, 0), tmpImg, 0.5, 0.5, cv2.INTER_LINEAR) 
        tmp = cv2.filter2D(tmpImg,-1,mask)
        gaussPyr.append(tmp.copy())
    return gaussPyr


def buildLaplacianPyramid(img, level):
    lapPyr = []  
    lapPyr.append(img.copy())
    tmpImg = img.copy()
    tmpPyr = img.copy()
    for i in range(1,level):
        cv2.resize(tmpImg, (0, 0), tmpImg, 0.5, 0.5, cv2.INTER_LINEAR)
        lapPyr.append(tmpImg.copy())
    for i in range(level - 1):
        cv2.resize(lapPyr[i + 1], (len(lapPyr[i][0]), len(lapPyr[i])), tmpPyr, 0, 0, cv2.INTER_LINEAR)
        cv2.subtract(lapPyr[i], tmpPyr)
    return lapPyr


def reconstructLaplacianPyramid(pyramid):
    level = len(pyramid)
    for i in range(level - 1,0):
        tmpPyr = cv2.resize(pyramid[i], (len(pyramid[0][0]),len(pyramid[0])),fx= 0,fy= 0,interpolation=cv2.INTER_LINEAR)
        pyramid[i - 1] = cv2.add(pyramid[i - 1], tmpPyr)
    return pyramid[0]


def fuseTwoImages(w1, img1, w2, img2, level):
    weight1 = buildGaussianPyramid(w1, level)
    weight2 = buildGaussianPyramid(w2, level)
    img1 = numpy.float32(img1)
    img2 = numpy.float32(img2)
    bgr = cv2.split(img1)
    bCnl1 = buildLaplacianPyramid(bgr[0], level)
    gCnl1 = buildLaplacianPyramid(bgr[1], level)
    rCnl1 = buildLaplacianPyramid(bgr[2], level)
    bgr = []
    bgr = cv2.split(img2)
    bCnl2 = buildLaplacianPyramid(bgr[0], level)
    gCnl2 = buildLaplacianPyramid(bgr[1], level)
    rCnl2 = buildLaplacianPyramid(bgr[2], level)
    bCnl=[]
    gCnl=[]
    rCnl=[]

    # TODO check the variable type here 
    # print("Type of bCnl1:", type(bCnl1))
    # print("Type of weight1:", type(weight1))

    for i in range(level):
        cn = cv2.add(cv2.multiply(bCnl1[i], weight1[i]), cv2.multiply(bCnl2[i], weight2[i]))
        bCnl.append(cn.copy())
        cn = cv2.add(cv2.multiply(gCnl1[i], weight1[i]), cv2.multiply(gCnl2[i], weight2[i]))
        gCnl.append(cn.copy())
        cn = cv2.add(cv2.multiply(rCnl1[i], weight1[i]), cv2.multiply(rCnl2[i], weight2[i]))
        rCnl.append(cn.copy())
    bChannel = reconstructLaplacianPyramid(bCnl)
    gChannel = reconstructLaplacianPyramid(gCnl)
    rChannel = reconstructLaplacianPyramid(rCnl)
    fusion = cv2.merge((bChannel, gChannel, rChannel))
    return fusion

def fuseThreeImages(w1, img1, w2, img2, w3, img3, level):
    weight1 = buildGaussianPyramid(w1, level)
    weight2 = buildGaussianPyramid(w2, level)
    weight3 = buildGaussianPyramid(w3, level)

    img1 = numpy.float32(img1)
    img2 = numpy.float32(img2)
    img3 = numpy.float32(img3)

    bgr = cv2.split(img1)
    bCnl1 = buildLaplacianPyramid(bgr[0], level)
    gCnl1 = buildLaplacianPyramid(bgr[1], level)
    rCnl1 = buildLaplacianPyramid(bgr[2], level)
    bgr = []
    bgr = cv2.split(img2)
    bCnl2 = buildLaplacianPyramid(bgr[0], level)
    gCnl2 = buildLaplacianPyramid(bgr[1], level)
    rCnl2 = buildLaplacianPyramid(bgr[2], level)
    bgr = cv2.split(img3)
    bCnl3 = buildLaplacianPyramid(bgr[0], level)
    gCnl3 = buildLaplacianPyramid(bgr[1], level)
    rCnl3 = buildLaplacianPyramid(bgr[2], level)

    bCnl=[]
    gCnl=[]
    rCnl=[]

    for i in range(level):
        cn = cv2.multiply(bCnl1[i], weight1[i]) + cv2.multiply(bCnl2[i], weight2[i]) + cv2.multiply(bCnl3[i], weight3[i])
        bCnl.append(cn.copy())
        cn = cv2.multiply(gCnl1[i], weight1[i]) + cv2.multiply(gCnl2[i], weight2[i]) + cv2.multiply(gCnl3[i], weight3[i])
        gCnl.append(cn.copy())
        cn = cv2.multiply(rCnl1[i], weight1[i]) + cv2.multiply(rCnl2[i], weight2[i]) + cv2.multiply(rCnl3[i], weight3[i])
        rCnl.append(cn.copy())
    bChannel = reconstructLaplacianPyramid(bCnl)
    gChannel = reconstructLaplacianPyramid(gCnl)
    rChannel = reconstructLaplacianPyramid(rCnl)
    fusion = cv2.merge((bChannel, gChannel, rChannel))
    return fusion

def fuseFourImages(w1, img1, w2, img2, w3, img3, w4, img4, level):
    weight1 = buildGaussianPyramid(w1, level)
    weight2 = buildGaussianPyramid(w2, level)
    weight3 = buildGaussianPyramid(w3, level)
    weight4 = buildGaussianPyramid(w4, level)

    img1 = numpy.float32(img1)
    img2 = numpy.float32(img2)
    img3 = numpy.float32(img3)
    img4 = numpy.float32(img4)

    bgr = cv2.split(img1)
    bCnl1 = buildLaplacianPyramid(bgr[0], level)
    gCnl1 = buildLaplacianPyramid(bgr[1], level)
    rCnl1 = buildLaplacianPyramid(bgr[2], level)
    bgr = []
    bgr = cv2.split(img2)
    bCnl2 = buildLaplacianPyramid(bgr[0], level)
    gCnl2 = buildLaplacianPyramid(bgr[1], level)
    rCnl2 = buildLaplacianPyramid(bgr[2], level)
    bgr = cv2.split(img3)
    bCnl3 = buildLaplacianPyramid(bgr[0], level)
    gCnl3 = buildLaplacianPyramid(bgr[1], level)
    rCnl3 = buildLaplacianPyramid(bgr[2], level)
    bgr = cv2.split(img4)
    bCnl4 = buildLaplacianPyramid(bgr[0], level)
    gCnl4 = buildLaplacianPyramid(bgr[1], level)
    rCnl4 = buildLaplacianPyramid(bgr[2], level)

    bCnl=[]
    gCnl=[]
    rCnl=[]

    # import pdb; pdb.set_trace()

    for i in range(level):
        cn = cv2.multiply(bCnl1[i], weight1[i]) + cv2.multiply(bCnl2[i], weight2[i]) + cv2.multiply(bCnl3[i], weight3[i]) + cv2.multiply(bCnl4[i], weight4[i])
        bCnl.append(cn.copy())
        cn = cv2.multiply(gCnl1[i], weight1[i]) + cv2.multiply(gCnl2[i], weight2[i]) + cv2.multiply(gCnl3[i], weight3[i]) + cv2.multiply(gCnl4[i], weight4[i])
        gCnl.append(cn.copy())
        cn = cv2.multiply(rCnl1[i], weight1[i]) + cv2.multiply(rCnl2[i], weight2[i]) + cv2.multiply(rCnl3[i], weight3[i]) + cv2.multiply(rCnl4[i], weight4[i])
        rCnl.append(cn.copy())
    bChannel = reconstructLaplacianPyramid(bCnl)
    gChannel = reconstructLaplacianPyramid(gCnl)
    rChannel = reconstructLaplacianPyramid(rCnl)
    fusion = cv2.merge((bChannel, gChannel, rChannel))
    return fusion