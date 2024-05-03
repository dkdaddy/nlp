import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
# %matplotlib inline

np.random.seed(42)
MAX_SEQUENCE_LENGTH=300

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)
wordindex=imdb.get_word_index(path="imdb_word_index.json")
wordindex = {k:(v+3) for k,v in wordindex.items()}
wordindex["<PAD>"] = 0
wordindex["<START>"] = 1
wordindex["<UNK>"] = 2
wordindex["<UNUSED>"] = 3

# textlengthsTrain=[len(t) for t in x_train]
# plt.hist(textlengthsTrain,bins=20)
# plt.title("Distribution of text lengths in words")
# plt.xlabel("number of words per document")
# plt.show()

inv_wordindex = {value:key for key,value in wordindex.items()}
print(' '.join(inv_wordindex[id] for id in x_train[1] ))
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
y_train =  keras.utils.to_categorical(np.asarray(y_train))
y_test =  keras.utils.to_categorical(np.asarray(y_test))
print('Shape of Training Data Input:', x_train.shape)
print('Shape of Training Data Labels:', y_train.shape)
print('Shape of Test Data Input:', x_test.shape)
print('Shape of Test Data Labels:', y_test.shape)

# print(x_test[1])
# print(y_test[1])


# words = x_train[0]
# vectorize_layer = TextVectorization(
#     max_tokens=5000,
#     output_mode='int',
#     output_sequence_length=300,
#     vocabulary=words)

# vectorize_layer.adapt(x_train)
# vectorize_layer.adapt(x_test)

# tokenizer = TextVectorization.Tokenizer(num_words=1000)
# x_train = vectorize_layer(x_train)
# x_test = vectorize_layer(x_test)
# print(x_train[0])

# num_classes = 2
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# print(y_train.shape)
# print(y_test.shape)

model = Sequential()
model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test), 
          verbose=2)

score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])

