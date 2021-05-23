import cv2 as cv
import numpy as np


def getImageFeature(imagePath, colorBins):
    img = cv.resize(cv.imread(imagePath), (255, 255))
    blur = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    h, _, _ = cv.split(cv.cvtColor(blur, cv.COLOR_BGR2HSV))
    hue = np.array(h).flatten()
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    filterImage = imageFilter(grayImage)
    hog = np.array(getHOGData(filterImage)).flatten()
    # cv.imshow('img', h)
    # cv.waitKey(0)
    counts, _ = np.histogram(hue, colorBins)
    nCounts = normalize(counts, 0, 1)
    feature = np.concatenate((nCounts, hog), axis=None)
    return feature


def imageFilter(grayImage):
    blur = cv.GaussianBlur(grayImage, (5, 5), 0)

    _, thresh1 = cv.threshold(blur, 135, 255, cv.THRESH_BINARY)

    filtY = np.array([[-1, 0, 1], [-1, 0, -1], [-1, 0, 1]])
    filtX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    outX = cv.filter2D(thresh1, -1, filtX, borderType=0)/255
    outY = cv.filter2D(thresh1, -1, filtY, borderType=0)/255
    out = np.sqrt((outX**2)+(outY**2))

    filt = np.ones((5, 5))
    outC = cv.morphologyEx(out, cv.MORPH_CLOSE, filt)

    return outC


def getHOGData(image):
    cv_img = image.astype(np.uint8)
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
    hist = hog.compute(cv_img, winStride, padding, locations)
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
