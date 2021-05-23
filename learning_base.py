import os
import pickle
from dotenv import load_dotenv
from sklearn.naive_bayes import GaussianNB
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
FEATURE_LABEL_FILE_NAME = os.getenv('FEATURE_LABEL_FILE_NAME')
MODEL_FILE_NAME = os.getenv('MODEL_FILE_NAME')


def main():
    # Extract image feature from dataset.
    features, labels = featureExtraction()
    #   Create Gaussian Naive Bayes.
    gnbModel = GaussianNB().fit(features, labels)
    #   Save model file.
    pickle.dump(gnbModel, open(MODEL_PATH + '/' + MODEL_FILE_NAME, 'wb'))
    #   Save features and labels file.
    pickle.dump(
        [features, labels],
        open(MODEL_PATH+'/'+FEATURE_LABEL_FILE_NAME, 'wb')
    )
    print('Done')


def featureExtraction():
    features = []
    labels = []
    colorBins = GetImageFeature.createBins(COLOR_BINS, COLOR_FEATURE_VALUE_MAX)
    # Search file in dataset.
    for PATH, _, _ in os.walk(DATA_SET_PATH):
        for filePath in os.listdir(PATH):
            if os.path.isfile(os.path.join(PATH, filePath)) and DATA_SET_FILE_TYPE in filePath:
                fullPath = '%s/%s' % (PATH, filePath)
                lable = PATH.split('/')[-1]
                features.append(
                    GetImageFeature.getImageFeature(fullPath, colorBins)
                )
                labels.append(lable)
        print(PATH)
    return [features, labels]


main()
