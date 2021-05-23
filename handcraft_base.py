import os
import json
import pickle
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
import GetImageFeature

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_FILE_NAME = os.getenv('MODEL_FILE_NAME')
TEST_FILE_PATH = os.getenv('TEST_FILE_PATH')
COLOR_FEATURE_VALUE_MAX = int(os.getenv('COLOR_FEATURE_VALUE_MAX'))
COLOR_BINS = int(os.getenv('COLOR_BINS'))


def main():
    gnbModel = pickle.load((open(MODEL_PATH+'/'+MODEL_FILE_NAME, 'rb')))
    testData = json.load(open(TEST_FILE_PATH))
    testFeatures = []
    testLabels = []
    for testObj in testData["data"]:
        feature = GetImageFeature.getImageFeature(
            testObj["imagePath"],
            GetImageFeature.createBins(COLOR_BINS, COLOR_FEATURE_VALUE_MAX)
        )
        testFeatures.append(feature)
        testLabels.append(testObj["label"])
    out = gnbModel.predict(testFeatures)
    result = accuracy_score(testLabels, out)
    print("Score: %.02f%%" % (result*100))
    print("Answer:")
    print(out)
    print("True:")
    print(testLabels)


main()
