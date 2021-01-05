
"""
This is is a part of the DeepLearning.AI TensorFlow Developer Professional Certificate offered on Coursera.

All copyrights belong to them. I am sharing this work here to showcase the projects I have worked on
Course: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

Week 3: Transfer Learning

Aim: Using the Inception V3 model
"""


import os
import tensorflow as tf
import numpy as np
import zipfile

from os import path, getcwd, chdir
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = "/Saved_models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

pre_trained_model = InceptionV3(input_shape=(150,150,3), include_top=False, weights=None)

for layer in pre_trained_model.layers:
    layer.trainable=False

last_layer = pre_trained_model.get_layer("mixed7")
print("last layer output shape: ", last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

x= layers.Flatten()(last_output)
x= layers.Dense(1024, activation="relu")(x)
x= layers.Dropout(0.2)(x)
x= layers.Dense(1, activation="sigmoid")(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["acc"])

model.summary()

"""Imported from cats & dogs small set"""

local_zip = f"{getcwd()}/Dataset/cats_and_dogs_filtered.zip"
zip_ref=zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/Dataset")
zip_ref.close()

base_dir = "/Dataset/cats_and_dogs_filtered"

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")

validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dogs_dir = os.path.join(validation_dir, "dogs")

train_cats_fnames = os.listdir(train_cats_dir)
train_dogs_fnames = os.listdir(train_dogs_dir)

print(train_cats_fnames[:10])
print(train_dogs_fnames[:10])

print("Total training cat images: ", len(os.listdir(train_cats_dir)))
print("Total training dog images: ", len(os.listdir(train_dogs_dir)))
print("Total Validation cat images: ", len(os.listdir(validation_cats_dir)))
print("Total Validation dog images: ", len(os.listdir(validation_dogs_dir)))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode="binary", target_size=(150,150))

validation_generator = train_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode="binary", target_size=(150,150))

history = model.fit_generator(train_generator, validation_data=validation_generator, steps_per_epoch=100, epochs= 5, validation_steps = 50)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

