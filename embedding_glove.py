import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt

N_EPOCHS = 10
###############################################################
# load data
###############################################################

imdb_dir = '../dataset/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

###############################################################
# tokenize the texts
###############################################################

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000 # top words in the dataset

tokenizer = Tokenizer(num_words=max_words)
# build the word index
tokenizer.fit_on_texts(texts)
# turn strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
# shuffle the indices, original data are in order
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_validation = data[training_samples: training_samples + validation_samples]
y_validation = labels[training_samples: training_samples + validation_samples]

###############################################################
# Glove word embedding
# 2014 English Wikipedia. 100-dimensional embedding vectors for
# 400000 words
###############################################################

glove_dir = '../dataset/glove.6B'
embedding_dim = 100
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.{}d.txt'.format(embedding_dim)))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found {} word vectors.'.format(len(embeddings_index)))

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        # if the word is not found, the embedding index will be all zeros as initial value
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    else:
        break

###############################################################
# define the model
###############################################################

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# load glove embeddings in the model
# Embedding layer has a single weight matrix 2D float matrix
# entry i and corresponding word vector.

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
H = model.fit(x_train, y_train,
              epochs=N_EPOCHS,
              batch_size=32,
              validation_data=(x_validation, y_validation))
model.save_weights('pre_trained_glove_model.h5')

###############################################################
# plot the training results
###############################################################

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N_EPOCHS,), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N_EPOCHS,), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N_EPOCHS,), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N_EPOCHS,), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

###############################################################
# evaluate the test data
###############################################################

test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

model.load_weights('pre_trained_glove_model.h5')
evaluation = model.evaluate(x_test, y_test)
print('loss: {}, metrics: {}'.format(evaluation[0], evaluation[1]))
