from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from pynlp.plot.historyplot import HistoryPlot

max_features = 10000
# max input length
maxlen = 20
N_epochs = 10

# (samples,)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# cut off the first 20 words from reviews, shape is (samples, 20)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Embedding layer has shape (samples, maxlen, 8), e.x. (None, 20, 8)
# It uses top 10000 most common words, cuts off the first 20 words from each review.
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
H = model.fit(x_train, y_train,
              epochs=N_epochs,
              batch_size=32,
              validation_split=0.2)

hp = HistoryPlot(N_epochs)
hp.show(H)
