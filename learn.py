import cv2 as cv
import numpy as np
import skimage.feature as skf
from sklearn import svm
import os
import matplotlib.pyplot as plt
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
                    fullPath, createBins(360, 360)))
                trainLables.append(lable)

    # clf = svm.SVC()
    # clf.fit(np.array(trainFeaetures), np.array(trainLables))
    gnb = GaussianNB().fit(trainFeaetures, trainLables)
    test = getFeature('./test/9-81.png', createBins(360, 360))
    # test = getFeature('./trainingData/Tr/9/81.jpg', createBins(360, 360))
    # print(len(test))
    # out = clf.predict([test])
    out = gnb.predict([test])
    print('Answer is ' + str(out))

    # test = ['./trainingData/Tr/9/81.jpg', './trainingData/Tr/1/1.jpg']
    # img1 = cv.cvtColor(cv.imread(test[0]), cv.COLOR_BGR2HSV)
    # imgH1, s1, v1 = cv.split(img1)
    # img2 = cv.cvtColor(cv.imread(test[0]), cv.COLOR_RGB2GRAY)
    # imgH2, s2, v2 = cv.split(img2)
    # cv.imshow('h1', v1)
    # cv.imshow('h2', img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # hue = np.array(imgH1).flatten()
    # plt.hist(hue)
    # plt.show()
    # plt.hist(hue*360, bins=360, range=(0.0, 360.0),
    #          histtype='stepfilled', color='r', label='Hue')
    # plt.legend()
    # plt.show()


def createBins(bins, max):
    myBins = []
    for i in range(1, bins+1):
        myBins.append((i * (max//bins))-1)
    return myBins


# print(createBins(360, 360))

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
    imgGray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    h, _, _ = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    h = np.array(h).flatten()
    imgOut = (imgGray / (256/64)).astype(int)
    glcm = skf.greycomatrix(
        imgOut, distances=[0], angles=[0, 45, 90, 135], levels=64, symmetric=True, normed=True)
    # [con] = skf.greycoprops(glcm, 'contrast')  # contrast
    # [dis] = skf.greycoprops(glcm, 'dissimilarity')  # dissimilarity
    # [hom] = skf.greycoprops(glcm, 'homogeneity')  # homogeneity
    [eng] = skf.greycoprops(glcm, 'energy')  # energy
    counts, _ = np.histogram(h, bins)
    # cMax = max(counts)
    # cMin = min(counts)
    norm = normalize(counts, 0, 1)
    print(eng)
    # max = max(counts)
    # min = min(counts)
    # print(counts)
    # print(len(counts))
    array = np.concatenate((np.array(eng), norm), axis=None)
    return array


main()
