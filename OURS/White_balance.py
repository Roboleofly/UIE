import cv2 as cv
import numpy as np


def White_balance(img):
    '''
    Gray World Assumption
    :param img: img:cv.imread
    :return: ima
    '''

    # Get the Blue, Green and Red channel of each pixel
    B = np.double(img[:, :, 0])
    G = np.double(img[:, :, 1])
    R = np.double(img[:, :, 2])

    # Calculate the mean value of each color channel
    B_aver = np.mean(B)
    G_aver = np.mean(G)
    R_aver = np.mean(R)

    # Calculate the w_1
    w_1 = (((B_aver - G_aver)/255.0)**2 + ((G_aver - R_aver)/255.0)**2 + ((B_aver - R_aver)/255.0)**2)

    # Get the corresponding proportion
    K = (B_aver + G_aver + R_aver ) / 3.0
    Kb = K / B_aver
    Kg = K / G_aver
    Kr = K / R_aver

    # Assign the new factor to the original color value
    Ba = ( B * Kb )
    Ga = ( G * Kg )
    Ra = ( R * Kr )

    # Check the value is reasonable
    for i in range (len(Ba)):
        for j in range (len(Ba[0])):
            if Ba[i][j] > 255:
                Ba[i][j] = 255
            if Ga[i][j] > 255:
                Ba[i][j] = 255
            if Ra[i][j] > 255:
                Ra[i][j] = 255

    # Here we can print the mean of each color channel
    #print(np.mean(Ba), np.mean(Ga), np.mean(Ra))

    color_corrected_img = np.uint8(np.zeros_like(img))

    if w_1 < 0.06:
        color_corrected_img[:, :, 0] = Ba
        color_corrected_img[:, :, 1] = Ga
        color_corrected_img[:, :, 2] = Ra
    else:
        color_corrected_img[:, :, 0] = 0.6 * B + 0.4 * Ba
        color_corrected_img[:, :, 1] = 0.6 * G + 0.4 * Ga
        color_corrected_img[:, :, 2] = 0.6 * R + 0.4 * Ra


    return color_corrected_img, w_1

if __name__ == '__main__':
    # Read the image
    img = cv.imread('data/raw/color_1.png')

    # Do the white balance
    enhanced_img, w1 = White_balance(img)
    print(enhanced_img)
    print(w1)
    res_img = np.hstack([img,enhanced_img])
    cv.imshow("Display window", res_img)

    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite("data/res/5.png", res_img)











