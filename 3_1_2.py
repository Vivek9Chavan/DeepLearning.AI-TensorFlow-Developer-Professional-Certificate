
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "Ich leibe mein Bulle",
    "Mein Bulle fickt mich sehr gut",
    "Ich komme jeder mal, wann mein Bulle mich fickt",
    "Er hat mich alle nacht gefickt",
    "Ich lecke sein Glied sehr gern",
    "Er macht mich nass, wann er mich berührt"
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
    "Ich bin so geil",
    "Leibe Bulle, komm, nimm mich, nimm meine muschi!"
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nPadded test sequence: ")
padded_test = pad_sequences(test_seq, maxlen=10)
print(padded_test)