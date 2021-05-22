import os
import cv2 as cv
import numpy as np
from sklearn.naive_bayes import GaussianNB


def main():
    """Main function"""
    # Set env
    config = {
        "startDir": './trainingData/Tr',
        "imageType": '.jpg'
    }
    trainFeaetures = []
    trainLables = []
    # Search file
    for path, _, _ in os.walk(config["startDir"]):
        for filePath in os.listdir(path):
            if os.path.isfile(os.path.join(path, filePath)) and config["imageType"] in filePath:
                fullPath = '%s/%s' % (path, filePath)
                lable = path.split('/')[-1]
                trainFeaetures.append(getFeature(
                    fullPath, createBins(100, 360)))
                trainLables.append(lable)

    gnb = GaussianNB().fit(trainFeaetures, trainLables)
    test = getFeature('./test/0-0.png', createBins(100, 360))
    # test = getFeature('./trainingData/Tr/4/31.jpg', createBins(100, 360))
    out = gnb.predict([test])
    print('Answer is ' + str(out))


def createBins(bins, max):
    myBins = []
    for i in range(1, bins+1):
        myBins.append((i * (max//bins))-1)
    return myBins


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def getFeature(imagePath, bins):
    """Useing GLCM for contrast, dissimilarity, homogeneity, energy"""
    img = cv.imread(imagePath)
    h, _, _ = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    h = np.array(h).flatten()
    hog = np.array(HOG(imagePath)).flatten()
    counts, _ = np.histogram(h, bins)
    norm = normalize(counts, 0, 1)
    array = np.concatenate((norm, hog), axis=None)
    return array


def HOG(path):
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
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                           histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)
    hist = hog.compute(image, winStride, padding, locations)
    return hist


main()
