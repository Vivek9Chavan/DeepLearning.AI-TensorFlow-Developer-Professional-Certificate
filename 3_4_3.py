"""
Tensorflow Tutorial!
https://www.tensorflow.org/tutorials/text/text_generation

"""
import tensorflow as tf
from os import getcwd
import numpy as np
import os
import time
from tensorflow.keras import Model


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))

# Take a look at the first 250 characters in text
print(text[:250])
