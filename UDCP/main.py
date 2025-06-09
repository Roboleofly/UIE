import os
import numpy as np
import cv2
import natsort

from .RefinedTramsmission import Refinedtransmission
from .getAtomsphericLight import getAtomsphericLight
from .getGbDarkChannel import getDarkChannel
from .getTM import getTransmission
from .sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')


def UDCP(img):
    blockSize = 9
    GB_Darkchannel = getDarkChannel(img, blockSize)
    AtomsphericLight = getAtomsphericLight(GB_Darkchannel, img)

    print('AtomsphericLight', AtomsphericLight)
    
    transmission = getTransmission(img, AtomsphericLight, blockSize)

    transmission = Refinedtransmission(transmission, img)
    sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

    return sceneRadiance

if __name__ == '__main__':

    img = cv2.imread('1.png')

    blockSize = 9
    GB_Darkchannel = getDarkChannel(img, blockSize)
    AtomsphericLight = getAtomsphericLight(GB_Darkchannel, img)

    print('AtomsphericLight', AtomsphericLight)
    # print('img/AtomsphericLight', img/AtomsphericLight)

    # AtomsphericLight = [231, 171, 60]

    transmission = getTransmission(img, AtomsphericLight, blockSize)

    cv2.imwrite('map_1.jpg', np.uint8(transmission * 255))

    transmission = Refinedtransmission(transmission, img)
    sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)
    # print('AtomsphericLight',AtomsphericLight)



    cv2.imwrite('transmission_1.jpg', np.uint8(transmission* 255))
    cv2.imwrite('seneradiance_1.png', sceneRadiance)


