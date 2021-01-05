
import os
import zipfile
from os import path, getcwd, chdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Parameters for plotting!

nrows = 4
ncols = 4

pic_index = 100

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()



model = tf.keras.Sequential([
    keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(300,300,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation= "relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss= "binary_crossentropy", metrics = ["acc"])

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

train_generator = train_datagen.flow_from_directory("/Dataset/horse-or-human/", target_size = (300,300), batch_size=128, class_mode = "binary")
validation_generator = validation_datagen.flow_from_directory("/Dataset/validation-horse-or-human/", target_size = (300,300), batch_size=32, class_mode = "binary")


history = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch=8, epochs=2)

model.save("Saved_models/1_4_2_model.h5")

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

