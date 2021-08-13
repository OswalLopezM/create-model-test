import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import logging
logging.basicConfig(
format="%(asctime)s%(levelname)s - %(message)s", level=logging.INFO
)


###################

#!/usr/bin/python

'''This script trains a 3-layer CNN on 5 classes of gym equipment photos
In this setup, we:
- put the A photos in images/data5/train/A
- put the A photos in images/data5/test/A
'''

# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras import backend as K
# import tensorflow as tf

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)



# dimensions of our images.n
img_width, img_height = 28, 28
#D:\Tesis\Applications\Hitting-Gym-with-NN\poundcake\images\data5\train
train_data_dir = './dataset/images_dataset_train_test/train'
validation_data_dir = './dataset/images_dataset_train_test/test'
nb_train_samples = 262007
nb_validation_samples = 112157
epochs = 10
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = ( img_width, img_height, 1)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(27,activation ="softmax"))

# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

#generate training data 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

#generate test data
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

#fit the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#save the model so it can easily be compiled later for predictions
# model.save_weights('minerva-50.h5')
model.save('minerva-10.h5')


#print the indices that Keras assigns to each class (need this for making predictions)
classes = train_generator.class_indices
print(classes)

logging.info("Classes: %s "%(classes))

logging.info("End creation model")