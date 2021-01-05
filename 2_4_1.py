
import csv
import os
import zipfile
from os import getcwd

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_zip = f"{getcwd()}/Dataset/archive.zip"
zip_ref=zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/Dataset/sign_mnist_train.csv")
local_zip=f"{getcwd()}/Dataset/validation-horse-or-human.zip"
zip_ref=zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/Dataset/sign_mnist_test.csv")
zip_ref.close()

def get_data (filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter =",")
        first_line = True
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                first_line=False
            else:
                temp_labels.append(row[0])
                image_data = row[1:]
                image_data_as_array=np.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
        images  = np.array(temp_images).astype("float")
        labels = np.array(temp_labels).astype("float")
    return images, labels


path_sign_mnist_train = f"{getcwd()}/Dataset/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/Dataset/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

training_images= np.expand_dims(training_images, axis=-1)
testing_images= np.expand_dims(testing_images, axis=-1)

train_datagen = ImageDataGenerator(rescale=1 / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1 / 255)

# Keep These
print(training_images.shape)
print(testing_images.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(26, activation='softmax')
])

batch_size = 32
# Compile Model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

"""
train_data=train_datagen.flow(training_images, training_labels, batch_size=batch_size)
test_data=validation_datagen.flow(testing_images, training_labels, batch_size=batch_size)

# Train the Model
history = model.fit_generator(train_data, validation_data=test_data, steps_per_epoch=len(train_data)/batch_size, epochs=2)
"""
history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=32),
                              steps_per_epoch=len(training_images) / 32,
                              epochs=50,
                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
                              validation_steps=len(testing_images) / 32)

model.evaluate(testing_images, testing_labels)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss =history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
