import os
import pickle
from dotenv import load_dotenv
import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
import GetImageFeature

# Load data from .env file.
load_dotenv()
DATA_SET_PATH = os.getenv('DATA_SET_PATH')
DATA_SET_FILE_TYPE = os.getenv('DATA_SET_FILE_TYPE')
DATA_TEST_PATH = os.getenv('DATA_TEST_PATH')
DATA_TEST_FILE_TYPE = os.getenv('DATA_TEST_FILE_TYPE')
COLOR_FEATURE_VALUE_MAX = int(os.getenv('COLOR_FEATURE_VALUE_MAX'))
COLOR_BINS = int(os.getenv('COLOR_BINS'))
MODEL_PATH = os.getenv('MODEL_PATH')
HAND_CRAFT_OUTPUT_FILE_NAME = os.getenv('HAND_CRAFT_OUTPUT_FILE_NAME')
MODEL_FILE_NAME = os.getenv('MODEL_FILE_NAME')


def handCraft():
    # Extract image feature from dataset.
    features, labels, _ = featureExtraction(DATA_SET_PATH, DATA_SET_FILE_TYPE)
    #   Create Gaussian Naive Bayes.
    svmModel = svm.SVC().fit(features, labels)
    #   Save model file.
    pickle.dump(svmModel, open(MODEL_PATH + '/hancraft_model_SVM.sav', 'wb'))
    #   Save features and labels file.
    pickle.dump([features, labels], open(
        MODEL_PATH + '/SVM_' + HAND_CRAFT_OUTPUT_FILE_NAME, 'wb'))
    print('Learn Done!')


def test():
    # Load model
    svmModel = pickle.load(
        (open(MODEL_PATH + '/hancraft_model_SVM.sav', 'rb')))
    testFeatures = []
    testLabels = []
    testFeatures, testLabels, fileName = featureExtraction(
        DATA_TEST_PATH, DATA_TEST_FILE_TYPE
    )
    out = svmModel.predict(testFeatures)
    testLabels = np.array(testLabels)
    # Calcurated accuracy.
    result = accuracy_score(testLabels, out)
    # Print result.
    for i in range(len(out)):
        print('Predict: %s\t, Answer: %s\t, Result: %s\t, File Name: %s\t' %
              (out[i], testLabels[i], out[i] in testLabels[i], fileName[i]))
    print("Accuracy: %.02f%%" % (result*100))
    print(metrics.classification_report(testLabels, out, digits=4))


def featureExtraction(dataPath, fileType):
    features = []
    labels = []
    fileName = []
    colorBins = GetImageFeature.createBins(COLOR_BINS, COLOR_FEATURE_VALUE_MAX)
    # Search file in dataset.
    for PATH, _, _ in os.walk(dataPath):
        for filePath in os.listdir(PATH):
            if os.path.isfile(os.path.join(PATH, filePath)) and fileType in filePath:
                fullPath = '%s/%s' % (PATH, filePath)
                lable = PATH.split('/')[-1]
                features.append(
                    GetImageFeature.getImageFeature(fullPath, colorBins)
                )
                labels.append(lable)
                fileName.append(fullPath)
        print(PATH)
    return [features, labels, fileName]


handCraft()
test()
