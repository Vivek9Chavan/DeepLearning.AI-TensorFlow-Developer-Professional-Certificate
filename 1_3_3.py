
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