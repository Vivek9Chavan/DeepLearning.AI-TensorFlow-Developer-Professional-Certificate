"""
This is is a part of the DeepLearning.AI TensorFlow Developer Professional Certificate offered on Coursera.

All copyrights belong to them. I am sharing this work here to showcase the projects I have worked on
Course: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

Week 1: Sentiment in text

Aim: Tokenization
"""

from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "I am visiting a friend tomorrow",
    "Morgen besuche ich ein Freund",
    "I am looking forward to the weekend",
    "My dog ate my homework"
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print("word index: ", word_index)
print(tokenizer.word_counts)
print(len(word_index))
