import tensorflow as tf
import numpy as np
import pathlib
import json

"""
dataset_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json"

sarcasm = tf.keras.utils.get_file("sarcasm.json", origin=dataset_url, untar=True)

print(len(sarcasm))
"""

with open("Dataset/sarcasm.json", "r") as file:
    datastore = json.load(file)

sentences = []
labels=[]
urls=[]

for item in datastore:
    sentences.append(item["headline"])
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

print("Length: ", len(word_index))
print("\nWord Index: ", word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding="post")

print(padded[0])
print(padded.shape)

