import os

import datetime
import numpy as np
import cv2
import natsort

from .GuidedFilter import GuidedFilter
from .backgroundLight import BLEstimation
from .depthMapEstimation import depthMap
from .depthMin import minDepth
from .getRGBTransmission import getRGBTransmissionESt
from .global_Stretching import global_stretching
from .refinedTransmissionMap import refinedtransmissionMap

from .sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')


def ULAP(img):
    blockSize = 9
    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    DepthMap = depthMap(img)
    DepthMap = global_stretching(DepthMap)
    guided_filter = GuidedFilter(img, gimfiltR, eps)
    refineDR = guided_filter.filter(DepthMap)
    refineDR = np.clip(refineDR, 0,1)

    #cv2.imwrite('OutputImages/' + str(i) + '_ULAPDepthMap.jpg', np.uint8(refineDR * 255))

    AtomsphericLight = BLEstimation(img, DepthMap) * 255

    d_0 = minDepth(img, AtomsphericLight)
    d_f = 8 * (DepthMap + d_0)
    transmissionB, transmissionG, transmissionR = getRGBTransmissionESt(d_f)

    transmission = refinedtransmissionMap(transmissionB, transmissionG, transmissionR, img)
    sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

    return sceneRadiance



if __name__ == '__main__':
    starttime = datetime.datetime.now()


    img = cv2.imread("../data/demo/1.jpg")

    blockSize = 9
    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    DepthMap = depthMap(img)
    DepthMap = global_stretching(DepthMap)
    guided_filter = GuidedFilter(img, gimfiltR, eps)
    refineDR = guided_filter.filter(DepthMap)
    refineDR = np.clip(refineDR, 0,1)

    #cv2.imwrite('OutputImages/' + str(i) + '_ULAPDepthMap.jpg', np.uint8(refineDR * 255))

    AtomsphericLight = BLEstimation(img, DepthMap) * 255

    d_0 = minDepth(img, AtomsphericLight)
    d_f = 8 * (DepthMap + d_0)
    transmissionB, transmissionG, transmissionR = getRGBTransmissionESt(d_f)

    transmission = refinedtransmissionMap(transmissionB, transmissionG, transmissionR, img)
    sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)


    #cv2.imwrite("Output_Images/L_t_20.jpg", np.uint8(transmission[:, :, 2] * 255))


    # print('AtomsphericLight',AtomsphericLight)

    cv2.imwrite("../data/demo/ULAP/L_1.jpg", sceneRadiance)

    Endtime = datetime.datetime.now()
    Time = Endtime - starttime
    print('Time', Time)


