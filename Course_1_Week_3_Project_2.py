
"""
This is is a part of the DeepLearning.AI TensorFlow Developer Professional Certificate offered on Coursera.

All copyrights belong to them. I am sharing this work here to showcase the projects I have worked on
Course: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

Week 1: A New Programming Paradigm

Aim: Convolutions, MaxPooling anc callbacks

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
            if(logs.get("acc")>0.90):
                print("\n Reached 90% accuracy, so stopping training!")
                self.model.stop_training=True


    callback1 = mycallback()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    model = tf.keras.models.Sequential([
        keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation="relu"),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, callbacks=[callback1])

    test_loss = model.evaluate(x_test, y_test)
    print("The test loss is: " + str(test_loss))

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, "acc")
    plot_graphs(history, "loss")
    """This is how it s done!"""

    model.save("1_3_3_model.h5")

    return history.epoch, history.history["acc"][-1]

_, _ = train_mnist()

new_model = tf.keras.models.load_model('Saved_models/1_3_3_model.h5')

# Check its architecture
new_model.summary()
