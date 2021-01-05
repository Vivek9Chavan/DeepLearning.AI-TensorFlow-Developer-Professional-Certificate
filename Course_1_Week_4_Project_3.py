"""
This is is a part of the DeepLearning.AI TensorFlow Developer Professional Certificate offered on Coursera.

All copyrights belong to them. I am sharing this work here to showcase the projects I have worked on
Course: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

Week 4: Using Real-world Images

Aim: Binary classification- Happy or sad faces
"""

import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

path = f"{getcwd()}/Dataset/happy-or-sad.zip"

zip_ref= zipfile.ZipFile(path, "r")
zip_ref.extractall("/Dataset/h-or-s")
zip_ref.close()

def train_happy_sad_model():

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get("acc")>0.999):
                print("\n 99.9% Stopping!")
                self.model.stop_training=True

    callback = myCallback()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), input_shape=(150,150,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss="binary_crossentropy", metrics=["acc"])

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    train_generator=train_datagen.flow_from_directory("/Dataset/h-or-s", target_size=(150,150), batch_size=64, class_mode="binary")

    history = model.fit(train_generator, epochs=20, callbacks=[callback])

    model.save("1_4_3_model.h5")

    import matplotlib.pyplot as plt

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        #plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        #plt.legend([string, 'val_' + string])
        plt.show()

    plot_graphs(history, "acc")
    plot_graphs(history, "loss")

    return history.history["acc"][-1]

train_happy_sad_model()

