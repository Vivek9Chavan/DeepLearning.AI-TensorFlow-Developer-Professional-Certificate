
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd
import zipfile

path_inception = f"{getcwd()}/Saved_models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Import the inception model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = path_inception

pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

# Print the model summary
#pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed3')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))

# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.97):
      print("\nReached 97% accuracy so cancelling training!")
      self.model.stop_training = True


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation = 'relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

model.summary()


"""From horse_human dataset"""

local_zip = f"{getcwd()}/Dataset/horse-or-human.zip"
zip_ref=zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/Dataset/horse-or-human")
local_zip=f"{getcwd()}/Dataset/validation-horse-or-human.zip"
zip_ref=zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("/Dataset/validation-horse-or-human")
zip_ref.close()

# Directory with our training horse pictures
train_horse_dir=os.path.join("/Dataset/horse-or-human/horses")

# Directory with our training human pictures
train_human_dir = os.path.join("/Dataset/horse-or-human/humans")

"""New Addition: Validation set"""

# Directory with our training horse pictures
validation_horse_dir=os.path.join("/Dataset/validation-horse-or-human/horses")

# Directory with our training human pictures
validation_human_dir = os.path.join("/Dataset/validation-horse-or-human/humans")

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_names = os.listdir(validation_horse_dir)
print(validation_horse_names[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

print("Total training horse images: ", len(os.listdir(train_horse_dir)))
print("Total training human images: ", len(os.listdir(train_human_dir)))
print("Total validation horse images: ", len(os.listdir(validation_horse_dir)))
print("Total validation human images: ", len(os.listdir(validation_human_dir)))

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory("/Dataset/horse-or-human/", target_size = (150,150), batch_size=128, class_mode = "binary")
validation_generator = validation_datagen.flow_from_directory("/Dataset/validation-horse-or-human/", target_size = (150,150), batch_size=32, class_mode = "binary")

# Run this and see how many epochs it should take before the callback
# fires, and stops training at 97% accuracy

callbacks = myCallback()
history = model.fit(train_generator, validation_data=validation_generator,
                             steps_per_epoch=50, epochs=5, validation_steps=50, verbose=0, callbacks=[callbacks])

import matplotlib.pyplot as plt
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'acc' ]
val_acc  = history.history[ 'val_acc' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss')
