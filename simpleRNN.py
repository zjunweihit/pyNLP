from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
import matplotlib.pyplot as plt
import numpy as np

# SimpleRNN, inputs of shape (batch_size, timesteps, input_features)
# if return_sequences=True, it returns the last output, whose shape is (batch_size, input_features)

max_features = 10000 # No. of words to consider as features
maxlen = 500
batch_size = 32
N_epochs = 10

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train_sequences')
print(len(input_test), 'test sequences')

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

model = Sequential()
model.add((Embedding(max_features, 32)))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
H = model.fit(input_train, y_train,
              epochs=N_epochs,
              batch_size=batch_size,
              validation_split=0.2)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N_epochs,), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N_epochs,), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N_epochs,), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N_epochs,), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
