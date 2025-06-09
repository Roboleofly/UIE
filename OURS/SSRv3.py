import numpy as np
import cv2 as cv

# This code contains the traditional Retinex algorithms

# Replace the zero value in the matrix
def replaceZeros(input):
    min_nonzero = min(input[np.nonzero(input)])
    input[input == 0] = min_nonzero
    return input

# SSR（Single Scale Retinex）
def SSRv3 (src_img, size, sigma):
    # Get the dimension of the input image
    dimensions = src_img.shape

    # Illumination map estimation
    row = dimensions[0]
    col = dimensions[1]
    # Gaussian Blur
    L_blur = cv.GaussianBlur(src_img, (size, size), sigma)
    img = replaceZeros(src_img)
    L_blur = replaceZeros(L_blur)

    # Normalized by log
    dst_Img = cv.log(img / 255.0)
    dst_Lblur = cv.log(L_blur / 255.0)

    # # Here we can add a weight factor
    # bright = src_img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    # bright_aver = np.sum(bright) / (row * col)
    # w_4 =  (255.0 - bright_aver) / 255.0

    # L(x,y) = S(x,y) * G(x,y)
    dst_IxL = cv.multiply(dst_Img, dst_Lblur)

    # Log R(x,y) = Log S(x,y) - Log L(x,y)
    log_R = cv.subtract(dst_Img, dst_IxL)

    # Normalized to 255
    dst_R = cv.normalize(log_R, None, 0, 255, cv.NORM_MINMAX)

    clip_range = (-1.5, 1.5)

    # 5. Clip the result to avoid over-enhancement in bright regions
    log_R = np.clip(log_R, clip_range[0], clip_range[1])


    img = cv.convertScaleAbs(dst_R)

    return img , 1.0 


if __name__ == '__main__':
    # Read the image
    img = cv.imread('1.png')

    # Do the enhancement by SSR
    sigma = 5.0
    enhanced_img, w_4 = SSRv3(img, 0, sigma)
    print('w4 = %f' %w_4)
    res_img = np.hstack([img, enhanced_img])
    window = cv.namedWindow("Display window", 0)
    cv.imshow("Display window", res_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


























