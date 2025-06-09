
# encoding=utf-8
import os
import numpy as np
import cv2
import natsort

from .LabStretching import LABStretching
from .color_equalisation import RGB_equalisation
from .global_stretching_RGB import stretching
from .relativeglobalhistogramstretching import RelativeGHstretching

np.seterr(over='ignore')

def RGHS(img):
    # sceneRadiance = RGB_equalisation(img)
    sceneRadiance = img
    # sceneRadiance = RelativeGHstretching(sceneRadiance, height, width)

    sceneRadiance = stretching(sceneRadiance)

    sceneRadiance = LABStretching(sceneRadiance)

    return sceneRadiance

if __name__ == '__main__':
    img = cv2.imread("../data/demo/5.png")

    height = len(img)
    width = len(img[0])
    # sceneRadiance = RGB_equalisation(img)

    sceneRadiance = img
    # sceneRadiance = RelativeGHstretching(sceneRadiance, height, width)

    sceneRadiance = stretching(sceneRadiance)

    sceneRadiance = LABStretching(sceneRadiance)

    cv2.imwrite('../data/demo/RGHS/R_5.png', sceneRadiance)
