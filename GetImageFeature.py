import cv2 as cv
import numpy as np
import PyEMD.BEMD as pe
import matplotlib.pyplot as plt


def testFilter():
    img = cv.cvtColor(
        cv.imread('./datatest/8-73-filter.jpg'), cv.COLOR_BGR2GRAY)
    # img = cv.cvtColor(
    #     cv.imread('./dataset/8/73.jpg'), cv.COLOR_BGR2GRAY)

    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # filtY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # filtX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # outX = cv.filter2D(thresh1, -1, filtX, borderType=0)/255
    # outY = cv.filter2D(thresh1, -1, filtY, borderType=0)/255
    # out = np.sqrt((outX**2)+(outY**2))

    # gaus = cv.getGaussianKernel(9, 3)
    # gauFilter = np.multiply(gaus.T, gaus)
    # outGau = cv.filter2D(out, -1, gauFilter, borderType=0)

    filt3 = np.ones((3, 3))
    outO = cv.morphologyEx(ret, cv.MORPH_OPEN, filt3)
    outC = cv.morphologyEx(outO, cv.MORPH_CLOSE, filt3)

    cv.imshow('img', thresh1)
    cv.waitKey(0)


testFilter()


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
