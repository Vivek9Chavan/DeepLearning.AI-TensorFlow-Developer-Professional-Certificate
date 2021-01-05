"""
This is is a part of the DeepLearning.AI TensorFlow Developer Professional Certificate offered on Coursera.
All copyrights belong to them. I am sharing this work here to showcase the projects I have worked on

Course: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

Week 1: A New Programming Paradigm

Aim: Predicting the y-axis values for the given x-axis values on a straight line:
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib as plt

xs = np.array([1, 2, 3, 4, 5, 6])
ys = np.array([1, 1.5, 2, 2.5, 3, 3.5])

model = tf.keras.Sequential([keras.layers.Dense(1, input_shape=[1])])

model.compile(optimizer="sgd", loss="mean_squared_error")

history = model.fit(xs, ys, epochs=50)

print(model.predict([9]))

"""
Plot MSE: The most simple configuration!
"""

import matplotlib.pyplot as plt

def plot_graphs(history,string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

plot_graphs(history,"loss")
