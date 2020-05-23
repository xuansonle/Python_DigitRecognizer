#!/usr/bin/env python
# coding: utf-8

### Load libraries 
import gzip
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Keras
from keras.datasets import mnist
import keras.models as models
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, UpSampling2D
from keras.utils import to_categorical
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, Adam

def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)

def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)


# Modelling

## Read data
X_train = open_images("./data/mnist/train-images-idx3-ubyte.gz")
y_train = open_labels("./data/mnist/train-labels-idx1-ubyte.gz")
X_test = open_images("./data/mnist/t10k-images-idx3-ubyte.gz")
y_test = open_labels("./data/mnist/t10k-labels-idx1-ubyte.gz")

#plt.imshow(X_train[1],cmap="gray_r")

## Prepare data
# One hot encoding: Multinomial: y = 0, y = 1, etc.
y_train_multi = to_categorical(y_train)
y_test_multi = to_categorical(y_test)

# Reshape input image (number of img, width in pixel, height in pixel, number of color layer)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Standard the input values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

## Build model
# Create the model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16, kernel_size=(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_multi, batch_size=32, epochs=5, verbose=1, validation_split=0.3)

# Evaluate on test data
model.evaluate(X_test, y_test_multi)

## Save model
model.save("model.h5")

