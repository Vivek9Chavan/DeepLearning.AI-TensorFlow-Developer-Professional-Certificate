"""
This is is a part of the DeepLearning.AI TensorFlow Developer Professional Certificate offered on Coursera.

All copyrights belong to them. I am sharing this work here to showcase the projects I have worked on
Course: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

Week 1: Sentiment in text

Aim: Tokenization and padding sequences
"""

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I am going to Berlin in April",
    "Berlin is a great city. The people are friendly!",
    "I plan to backpack through Europe",
    "This year is going to bring a 1ot of changes",
    "I am learning TensorFlow"
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, maxlen=7)
print("\nWord Index: ", word_index)
print("\nSequences: ", sequences)
print("\nPadded Sequences: ")
print(padded)

# Try with words that the tokenizer wasn't fit to
test_data=[
    "I am so happy!",
    "Happy new year 2021!!"
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nPadded test sequence: ")
padded_test = pad_sequences(test_seq, maxlen=10)
print(padded_test)
