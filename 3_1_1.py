
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "Ich leibe mein Bulle",
    "Mein Bulle fickt mich sehr gut",
    "Ich komme jeder mal, wann mein Bulle mich fickt",
    "Er hat mich alle nacht gefickt"
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print("word index: ", word_index)
print(tokenizer.word_counts)
print(len(word_index))