import cv2 as cv
import numpy as np
from PyEMD.EMD2d import EMD2D


def testBEMD():
    img = cv.imread('./datatest/lena.png')
    g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    emd2d = EMD2D()
    IMFs_2D = emd2d(g_img)
    print(len(IMFs_2D))
    cv.imshow('IMF0', IMFs_2D[0])
    cv.imshow('IMF1', IMFs_2D[1])
    cv.waitKey(0)


# testBEMD()


def getImageFeature(imagePath, colorBins):
    img = cv.imread(imagePath)
    h, _, _ = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    h = np.array(h).flatten()
    hog = np.array(getHOGData(imagePath)).flatten()
    counts, _ = np.histogram(h, colorBins)
    nCounts = normalize(counts, 0, 1)
    feature = np.concatenate((nCounts, hog), axis=None)
    return feature


def getHOGData(path):
    image = cv.imread(path, 0)
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv.HOGDescriptor(
        winSize,
        blockSize,
        blockStride,
        cellSize,
        nbins,
        derivAperture,
        winSigma,
        histogramNormType,
        L2HysThreshold,
        gammaCorrection,
        nlevels
    )
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)
    hist = hog.compute(image, winStride, padding, locations)
    return hist


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def createBins(bins, max):
    myBins = []
    for i in range(0, bins+1):
        myBins.append((i * (max//bins)))
    return myBins
