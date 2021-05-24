import os
from dotenv import load_dotenv

import cv2 as cv
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load data from .env file.
load_dotenv()
DATA_SET_PATH = os.getenv('DATA_SET_PATH')
DATA_SET_FILE_TYPE = os.getenv('DATA_SET_FILE_TYPE')

features = []
labels = []

# get data
for PATH, _, _ in os.walk(DATA_SET_PATH):
    for filePath in os.listdir(PATH):
        if os.path.isfile(os.path.join(PATH, filePath)) and DATA_SET_FILE_TYPE in filePath:
            fullPath = '%s/%s' % (PATH, filePath)
            lable = PATH.split('/')[-1]
            img = imread(fullPath)
            img = cv.resize(img, (64, 64))

            # normalizing the pixel values
            # img /= 255.0
            # converting the type of pixel to float
            # img = img.astype('float32')
            # appending the image into the list
            features.append(
                img
            )
            labels.append(int(lable))

train_x, test_x, train_y, test_y = train_test_split(
    features, labels, test_size=0.25)

train_x = np.array(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y)
test_y = np.array(test_y)

# images are greyscale, they have 1 channel
# add dim to (n_samples, height, width, channels)
train_y = tf.expand_dims(train_y, axis=-1)
test_y = tf.expand_dims(test_y, axis=-1)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

cnn = models.Sequential([
    layers.Conv2D(filters=64, kernel_size=(3, 3),
                  activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(13, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(train_x, train_y, epochs=12)

print('-----')
print('test')
cnn.evaluate(test_x, test_y)
