import cv2 as cv
import numpy as np
import skimage.feature as skf
paths = ['./trainingData/Tr/1/1.jpg', './trainingData/Tr/1/2.jpg']

for path in paths:
    img = cv.imread(path)
    imgGray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    glcm = skf.greycomatrix(
        imgGray, distances=[0], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)

    con = skf.greycoprops(glcm, 'contrast')  # contrast
    dis = skf.greycoprops(glcm, 'dissimilarity')  # dissimilarity
    hom = skf.greycoprops(glcm, 'homogeneity')  # homogeneity
    ent = skf.greycoprops(glcm, 'energy')  # energy

    print(con)
    print(dis)
    print(hom)
    print(ent)
