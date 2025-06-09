import numpy as np
import cv2 as cv

# This is a demo to figure the dehaze algorithm
def DeHaze(img):
    # Get the dimension of the input image
    dimensions = img.shape

    # Illumination map estimation
    row = dimensions[0]
    col = dimensions[1]

    # print(img)
    # Find the minimum of the r, g, b value
    # V1 = min J^c(y)
    V1 = np.min(img, 2)
    # J^Dark = min V1
    # Here the patch size is set as 15x15
    Dark_channel = cv.erode(V1, np.ones((15, 15)))

    # Calculate the w_2
    #print(V1)
    t_sum = np.sum(V1 / 255.0)
    #print("The sum of the dark channel = %f " % t_sum)
    w_2 = t_sum / (row * col)
    w_2 = 0.4 * w_2                   # Here multiple the factor 0.2
    #print("The weight 2  value = %f " % w_2)

    # show the dark channel image
    # cv.imshow("Dark_channel", Dark_channel)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Estimate the Atmospheric Light A
    # We first pick the top 0.1 percent brightest pixels in the dark channel
    # The pixels with highest intensity in the input image I are selected as the A
    # Create the histogram of the dark channel and the bins value is the number of bin
    bins = 2000
    ht = np.histogram(V1, bins)

    # Select the top value
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(img, 2)[V1 >= ht[1][lmax]].max()


    # Estimate the transmission t
    # t = 1 - min min I^c(y) / A^c
    w = 0.95
    t = 1 - w * (Dark_channel / A)


    # REstore the image
    # J = ( I(x) - A ) / ( max (t, t0)) + A
    r_channel = (img[:, :, 0] - A) / t + A
    g_channel = (img[:, :, 1] - A ) / t + A
    b_channel = (img[:, :, 2] - A ) / t + A

    enhanced_img = np.ones(img.shape)
    enhanced_img[:, :, 0] = r_channel
    enhanced_img[:, :, 1] = g_channel
    enhanced_img[:, :, 2] = b_channel

    # Here may be some problem
    enhanced_img = enhanced_img.astype(np.uint8)

    #print(img.dtype)
    #print(enhanced_img.dtype)

    return enhanced_img, w_2


def dehaze(image, omega=0.95, t_min=0.1, window_size=15):
    # 归一化
    image = image.astype('float64') / 255
    # 求暗通道
    min_channel = np.amin(image, axis=2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (window_size, window_size))
    dark_channel = cv.erode(min_channel, kernel)
    # 估计大气光
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    search_idx = (-flat_dark).argsort()[:int(flat_dark.size * 0.001)]
    A = np.mean(flat_image[search_idx], axis=0)
    # 计算透射率
    transmission = 1 - omega * (dark_channel / A.max())
    # 限制透射率
    transmission = np.clip(transmission, t_min, 1)
    # 复原图像
    transmission = cv.resize(transmission, (image.shape[1], image.shape[0]))
    restored = np.empty_like(image)
    for i in range(3):
        restored[:, :, i] = (image[:, :, i] - A[i]) / transmission + A[i]
    restored = np.clip(restored, 0, 1)
    restored = (restored * 255).astype('uint8')
    return restored



if __name__ == '__main__':
    img = cv.imread('data/raw/haze_1.png')
    print(img)
    enhanced_img, w2 = DeHaze(img)
    print(enhanced_img)
    print(w2)
    res_img = np.hstack([img, enhanced_img])
    window = cv.namedWindow("Display window", 0)
    cv.imshow("Display window", res_img/255.0)
    cv.waitKey(0)




