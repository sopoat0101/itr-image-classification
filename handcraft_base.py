import os
import json
import pickle
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import accuracy_score
import GetImageFeature

# Load data from .env file.
load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_FILE_NAME = os.getenv('MODEL_FILE_NAME')
TEST_FILE_PATH = os.getenv('TEST_FILE_PATH')
COLOR_FEATURE_VALUE_MAX = int(os.getenv('COLOR_FEATURE_VALUE_MAX'))
COLOR_BINS = int(os.getenv('COLOR_BINS'))


def main():
    # Load model
    gnbModel = pickle.load((open(MODEL_PATH+'/'+MODEL_FILE_NAME, 'rb')))
    # Load test json data.
    testData = json.load(open(TEST_FILE_PATH))
    testFeatures = []
    testLabels = []
    # Set features and labels from json data.
    for testObj in testData["data"]:
        feature = GetImageFeature.getImageFeature(
            testObj["imagePath"],
            GetImageFeature.createBins(COLOR_BINS, COLOR_FEATURE_VALUE_MAX)
        )
        testFeatures.append(feature)
        testLabels.append(testObj["label"])
    out = gnbModel.predict(testFeatures)
    # Calcurated accuracy.
    result = accuracy_score(testLabels, out)
    # Print result.
    print("Accuracy: %.02f%%" % (result*100))
    print("Answer:")
    print(out)
    print("Correct:")
    print(np.array(testLabels))


main()
