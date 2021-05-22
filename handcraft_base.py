import os
import json
import pickle
from dotenv import load_dotenv
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
    result = gnbModel.score(testFeatures, testLabels)
    print("Scroe: %.02f%%" % (result*100))
    out = gnbModel.predict(testFeatures)
    print("Answer:")
    print(out)


main()
