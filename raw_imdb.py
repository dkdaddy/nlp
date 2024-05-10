import numpy as np
# import tensorflow as tf
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

from html_format import format_page

np.random.seed(42)
MAX_SEQUENCE_LENGTH=1000
train_sample=25000
test_sample=5000
x_train_neg = open("aclimdb/train_neg_reviews.txt").readlines()[0:train_sample]
x_train_pos = open("aclimdb/train_pos_reviews.txt").readlines()[0:train_sample]
x_train = keras.ops.convert_to_tensor(x_train_neg + x_train_pos, dtype="string")
y_train = keras.utils.to_categorical([0]*len(x_train_neg) + [1]*len(x_train_pos), 2)

x_test_neg = open("aclimdb/test_neg_reviews.txt").readlines()[0:test_sample]
x_test_pos = open("aclimdb/test_pos_reviews.txt").readlines()[0:test_sample]
x_test = keras.ops.convert_to_tensor(x_test_neg + x_test_pos, dtype="string")
y_test = keras.utils.to_categorical([0]*len(x_test_neg) + [1]*len(x_test_pos), 2)

print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

vectorize_layer = TextVectorization(
    max_tokens=1000,
    standardize="lower_and_strip_punctuation",
    output_mode='multi_hot')
    # output_sequence_length=100)

vectorize_layer.adapt(x_train)
# vectorize_layer.adapt(x_test)
vocab = vectorize_layer.get_vocabulary()
print(f"{len(vocab)}")
# print(x_train)
 
input_data = [["on the grand staircase"], ["read the clear and simple"]]
data = vectorize_layer(input_data)
print(data)
# https://stackoverflow.com/questions/78034587/textvectorization-issue
# https://keras.io/api/layers/core_layers/input/
model = keras.models.Sequential()
model.add(keras.Input(shape=(1,), dtype="string")) 
model.add(vectorize_layer) 
# model.add(keras.layers.Dense(4, activation='relu'))
# model.add(keras.layers.Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# model.predict(keras.ops.convert_to_tensor(["a"], dtype="string"))

hist = model.fit(x_train, y_train,
          batch_size=16,
          epochs=4,
          validation_data=(x_test, y_test), 
          verbose=2)

score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])
bad = model.predict(keras.ops.convert_to_tensor(x_test_neg))
good = model.predict(keras.ops.convert_to_tensor(x_test_pos))

good_err = [ (x[0], x[1], i, x_test_pos[i]) for i, x in enumerate(good) if x[0]>x[1]]
bad_err = [ (x[0], x[1], i, x_test_neg[i]) for i, x in enumerate(bad) if x[0]<x[1]]

very_good = [ (x[0], x[1], i, x_test_pos[i]) for i, x in enumerate(good) if x[1]>.99]
very_bad = [ (x[0], x[1], i, x_test_neg[i]) for i, x in enumerate(bad) if x[0]>.99]

print("bad classified as good", len(bad_err))
# for item in bad_err:
#     print(item)
print("\n\n\ngood classified as bad", len(good_err))
# for item in good_err:
#     print(item)
print("\n\n\nvery good", len(very_good))
# for item in very_good:
#     print(item)
print("\n\n\nvery bad", len(very_bad))
# for item in very_bad:
#     print(item)
# This is a logistic regression in Keras
# x = Input(shape=(32,))
# y = Dense(16, activation='softmax')(x)
# model = Model(x, y)
# print(very_good[0])
# print(very_bad[0])

data = [
    [.05, .95, "great movie blah blah blah CORRECT TP", True],
    [.65, .35, "terrible dah dah CORRECT TN", False],
    [.15, .85, "oh no dah dah WRONG FP", False],
    [.75, .25, "good dah dah WRONG FN", True]
]
data = [ (x[0], x[1], x_test_pos[i], True) for i, x in enumerate(good)] + \
       [ (x[0], x[1], x_test_neg[i], False) for i, x in enumerate(bad)] 
data.sort(key= lambda x: x[0])
format_page(data)