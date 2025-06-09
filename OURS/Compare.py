import cv2
import cv2 as cv
import numpy as np
import os

if __name__ == "__main__":
    file_path_name_1 = "Temp_1"
    file_path_name_2 = "Temp_2"
    print("Start")
    for file_name in os.listdir(file_path_name_1):
        img_1 = cv.imread(file_path_name_1+'/'+file_name)
        img_2 = cv.imread(file_path_name_2+'/'+file_name)
        comp_img = np.hstack([img_1, img_2])
        comp_img = cv2.resize(comp_img,(1024,512))
        #cv.imshow('Compare', comp_img)
        cv.imwrite('Com_res/' + file_name, comp_img)
        #cv.waitKey(0)




