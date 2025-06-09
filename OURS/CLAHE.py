import cv2
import cv2 as cv
import numpy as np

def CLAHE(img):
    # split the three channel
    B, G, R = cv.split(img)

    img_grey = cv.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bins = 100
    ht = np.histogram(img_grey, bins)

    # Select the top value
    d = np.cumsum(ht[0]) / float(img_grey.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.8:
            break
    v_max = ht[1][lmax]

    for lmin in range(0, bins, 1):
        if d[lmin] >= 0.2:
            break
    v_min = ht[1][lmin]

    w_3 = ( v_max - v_min ) / 255.0


    # Create CLAHE instance
    clahe = cv.createCLAHE(clipLimit=2.0)

    # Apply into each channel
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)

    # Merge into a RGB image
    clahe_img = cv.merge((clahe_B, clahe_G, clahe_R))

    return clahe_img , w_3




if __name__ == '__main__':
    img = cv.imread("data/raw/contrast_3.jpg")
    # Show the result of the image
    enhanced_img, w3 = CLAHE(img)
    print(enhanced_img)
    print(w3)
    res_img = np.hstack([img, enhanced_img])
    window = cv.namedWindow("Display window", 0)
    cv.imshow("Display window", res_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
