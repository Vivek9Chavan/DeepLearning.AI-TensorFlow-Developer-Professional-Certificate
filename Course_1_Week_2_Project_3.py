
"""
This is is a part of the DeepLearning.AI TensorFlow Developer Professional Certificate offered on Coursera.

All copyrights belong to them. I am sharing this work here to showcase the projects I have worked on
Course: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

Week 1: Introduction to Computer Vision

Aim: Implementing Callback
"""
"""
Exercise 2
In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:

It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
If you add any additional variables, make sure you use the same names as the ones used in the class
I've started the code for you below -- how would you finish it?
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import path, getcwd, chdir

path=f"{getcwd()}/Dataset/mnist.npz"

def train_mnist():
    mnist =tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) =mnist.load_data(path=path)
    x_train=x_train/255
    x_test=x_test/255

    class mycallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get("acc")>0.99):
                print("\n Reached 99% accuracy, so stopping training!")
                self.model.stop_training=True


    callback1 = mycallback()

    model = tf.keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[callback1])

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, "acc")
    plot_graphs(history, "loss")

    return history.epoch, history.history["acc"][-1]

train_mnist()

