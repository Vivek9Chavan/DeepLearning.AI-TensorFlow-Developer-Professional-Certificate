"""
This is is a part of the DeepLearning.AI TensorFlow Developer Professional Certificate offered on Coursera.
All copyrights belong to them. I am sharing this work here to showcase the projects I have worked on

Course: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

Week 2: Introduction to Computer Vision

Aim: Predicting the y-axis values for the given x-axis values on a straight line:
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

plt.imshow(training_images[0])
plt.show()

training_images=training_images/255
testing_images=testing_images/255

model= tf.keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

history = model.fit(training_images, training_labels, validation_data=(testing_images, testing_labels), epochs=5)

"""model.evaluate(training_images, training_labels)"""

def plot_graphs(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history,"acc")
plot_graphs(history,"loss")




