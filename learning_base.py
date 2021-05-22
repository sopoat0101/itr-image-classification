import os
import pickle
from dotenv import load_dotenv
from sklearn.naive_bayes import GaussianNB
import GetImageFeature

load_dotenv()
DATA_SET_PATH = os.getenv('DATA_SET_PATH')
DATA_SET_FILE_TYPE = os.getenv('DATA_SET_FILE_TYPE')
DATA_TEST_PATH = os.getenv('DATA_TEST_PATH')
DATA_TEST_FILE_TYPE = os.getenv('DATA_TEST_FILE_TYPE')
COLOR_FEATURE_VALUE_MAX = int(os.getenv('COLOR_FEATURE_VALUE_MAX'))
COLOR_BINS = int(os.getenv('COLOR_BINS'))
MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_FILE_NAME = os.getenv('MODEL_FILE_NAME')


def main():
    features, labels = featureExtraction()
    gnbModel = GaussianNB().fit(features, labels)
    pickle.dump(gnbModel, open(MODEL_PATH + '/' + MODEL_FILE_NAME, 'wb'))


def featureExtraction():
    features = []
    labels = []
    colorBins = GetImageFeature.createBins(COLOR_BINS, COLOR_FEATURE_VALUE_MAX)
    for PATH, _, _ in os.walk(DATA_SET_PATH):
        for filePath in os.listdir(PATH):
            if os.path.isfile(os.path.join(PATH, filePath)) and DATA_SET_FILE_TYPE in filePath:
                fullPath = '%s/%s' % (PATH, filePath)
                lable = PATH.split('/')[-1]
                features.append(
                    GetImageFeature.getImageFeature(fullPath, colorBins)
                )
                labels.append(lable)
    return [features, labels]


main()
