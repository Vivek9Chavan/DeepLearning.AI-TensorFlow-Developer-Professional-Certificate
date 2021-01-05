

"""
Cats vs Dogs
"""

import os
import pathlib
import zipfile
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import random

from shutil import copyfile
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import path, getcwd, chdir

local_zip = f"{getcwd()}/Dataset/cats-and-dogs.zip"
#shutil.rmtree("/tmp")

zip_ref=zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/Dataset/cats-and-dogs")
zip_ref.close()

print(len(os.listdir("/Dataset/cats-and-dogs/PetImages/Cat")))
print(len(os.listdir("/Dataset/cats-and-dogs/PetImages/Dog")))

try:
    os.mkdir("/Dataset/cats_v_dogs")
    os.mkdir("/Dataset/cats_v_dogs/training")
    os.mkdir("/Dataset/cats_v_dogs/testing")
    os.mkdir("/Dataset/cats_v_dogs/training/cats")
    os.mkdir("/Dataset/cats_v_dogs/training/dogs")
    os.mkdir("/Dataset/cats_v_dogs/testing/cats")
    os.mkdir("/Dataset/cats_v_dogs/testing/dogs")
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + "is zero length, so ignoring!")

    training_length = int(len(files)*SPLIT_SIZE)
    testing_length = int(len(files) - training_length)

    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = "/Dataset/PetImages/Cat/"
TRAINING_CATS_DIR = "/Dataset/cats_v_dogs/training/cats/"
TESTING_CATS_DIR = "/Dataset/cats_v_dogs/testing/cats/"
DOG_SOURCE_DIR = "/Dataset/PetImages/Dog/"
TRAINING_DOGS_DIR = "/Dataset/cats_v_dogs/training/dogs/"
TESTING_DOGS_DIR = "/Dataset/cats_v_dogs/testing/dogs/"

split_size = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir("/Dataset/cats_v_dogs/training/cats")))
print(len(os.listdir("/Dataset/cats_v_dogs/training/dogs")))
print(len(os.listdir("/Dataset/cats_v_dogs/testing/cats")))
print(len(os.listdir("/Dataset/cats_v_dogs/testing/cats")))

model = tf.keras.models.Sequential([
# YOUR CODE HERE
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

training_dir = "/Dataset/cats_v_dogs/training"
validation_dir = "/Dataset/cats_v_dogs/testing"

train_datagen = ImageDataGenerator(rescale=1/255, rotation_range=40, height_shift_range=0.2, width_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(training_dir, batch_size = 16, class_mode="binary", target_size=(150,150))
validation_generator = train_datagen.flow_from_directory(validation_dir, batch_size = 16, class_mode="binary", target_size=(150,150))

model.fit_generator(train_generator, validation_data=validation_generator, epochs= 2)

