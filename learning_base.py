import os
import cv2 as cv
import numpy as np
from dotenv import load_dotenv
from sklearn.naive_bayes import GaussianNB

load_dotenv()
DATA_SET_PATH = os.getenv('DATA_SET_PATH')
DATA_SET_FILE_TYPE = os.getenv('DATA_SET_FILE_TYPE')
DATA_TEST_PATH = os.getenv('DATA_TEST_PATH')
DATA_TEST_FILE_TYPE = os.getenv('DATA_TEST_FILE_TYPE')
COLOR_FEATURE_VALUE_MAX = int(os.getenv('COLOR_FEATURE_VALUE_MAX'))
COLOR_BINS = int(os.getenv('COLOR_BINS'))


def main():
    features, labels = featureExtraction()
    gnb = GaussianNB().fit(features, labels)
    test = getImageFeature(
        './dataset/1/1.jpg', createBins(COLOR_BINS, COLOR_FEATURE_VALUE_MAX))
    out = gnb.predict([test])
    print('Answer is ' + str(out))


def featureExtraction():
    features = []
    labels = []
    colorBins = createBins(COLOR_BINS, COLOR_FEATURE_VALUE_MAX)
    for PATH, _, _ in os.walk(DATA_SET_PATH):
        for filePath in os.listdir(PATH):
            if os.path.isfile(os.path.join(PATH, filePath)) and DATA_SET_FILE_TYPE in filePath:
                fullPath = '%s/%s' % (PATH, filePath)
                lable = PATH.split('/')[-1]
                features.append(getImageFeature(fullPath, colorBins))
                labels.append(lable)
    return [features, labels]


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


main()
