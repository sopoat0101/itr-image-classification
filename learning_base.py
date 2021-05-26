import os
from dotenv import load_dotenv
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data from .env file.
load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')
DATA_SET_PATH = os.getenv('DATA_SET_PATH')
DATA_TEST_PATH = os.getenv('DATA_TEST_PATH')
DATA_SET_FILE_TYPE = os.getenv('DATA_SET_FILE_TYPE')
LEARNNING_OUTPUT_FILE_NAME = os.getenv('LEARNNING_OUTPUT_FILE_NAME')


def train():

    train_x, train_y = getImagesAndLabels(DATA_SET_PATH)
    train_y = tf.expand_dims(train_y, axis=-1)

    cnn = models.Sequential([
        # Feature Ex
        layers.Conv2D(filters=32, kernel_size=(3, 3),
                      activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Classification
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(12, activation='softmax')
    ])

    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    cnn.fit(train_x, train_y, epochs=12)
    cnn.save((MODEL_PATH+'/'+LEARNNING_OUTPUT_FILE_NAME))


def test():
    cnn = models.load_model((MODEL_PATH+'/'+LEARNNING_OUTPUT_FILE_NAME))
    test_x, test_y = getImagesAndLabels(DATA_TEST_PATH)
    test_y = tf.expand_dims(test_y, axis=-1)
    print('---Test---')
    cnn.evaluate(test_x, test_y)


def getImagesAndLabels(filePath):
    images = []
    labels = []
    for path, _, _ in os.walk(filePath):
        for filePath in os.listdir(path):
            if os.path.isfile(os.path.join(path, filePath)) and DATA_SET_FILE_TYPE in filePath:
                fullPath = '%s/%s' % (path, filePath)
                lable = path.split('/')[-1]
                img = cv.imread(fullPath)
                img = cv.resize(img, (64, 64))
                images.append(img)
                labels.append(int(lable))
    return (np.array(images), np.array(labels))
