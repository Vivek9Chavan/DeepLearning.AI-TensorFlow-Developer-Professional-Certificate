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

"""

xs = np.array([1, 2, 3, 4, 5, 6])
ys = np.array([1, 1.5, 2, 2.5, 3, 3.5])

model=tf.keras.Sequential([keras.layers.Dense(1, input_shape=[1])])

model.compile(optimizer="sgd", loss= "mean_squared_error")

model.fit(xs, ys, epochs=500)

print(model.predict([7]))
"""

def house_model(y_new):
    xs = np.array([1, 2, 3, 4, 5, 6])
    ys = np.array([1, 1.5, 2, 2.5, 3, 3.5])
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss= 'mean_squared_error')
    model.fit(xs, ys, epochs=5000)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)
