import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from pynlp.plot.historyplot import HistoryPlot


data_dir = '../dataset/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))

# Not include date in the first column
weather_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    weather_data[i, :] = values
#temp = weather_data[:, 1]
#plt.plot(range(len(temp)), temp)
#plt.plot(range(1400), temp[:1400]) # show the 10 days data
#plt.show()

# normalize the data
mean = weather_data[:200000].mean(axis=0)
weather_data -= mean
std = weather_data[:200000].std(axis=0)
weather_data /= std

# min_index: the start index, including prev range
# max_index: the end index, including next range
# step 6, a step is 10 minutes, step 6 is 1 hour.
def generator(data, prev, next, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    # It's None for the last dataset, which will get the last "next" range data from the end
    if max_index is None:
        max_index = len(data) - next - 1 # index starts from 0
    i = min_index + prev # start from next index from "prev" range data
    while True:
        if shuffle:
            # get the batch_size data from start to end, some of them may be same
            rows = np.random.randint(min_index + prev, max_index, size=batch_size)
        else:
            if i + batch_size > max_index:
                i = min_index + prev
            rows = np.arange(i, min(i+batch_size, max_index))

        # samples' shape: (batch_size, time(hours), data)
        # deal with batch size each time, 240 hours, 10 days, weather data(14)
        samples = np.zeros((len(rows),
                            prev // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            # indices' step is per-day (6 * 10 min)
            indices = range(rows[j] - prev, rows[j], step)
            samples[j] = data[indices]
            # next day temperature (T (degC))
            targets[j] = data[rows[j] + next][1]
        yield samples, targets

prev = 1440 #10 days
step = 6 # 1 hour
next = 144 # predict next day
batch_size = 128

train_data = generator(weather_data,
                       prev=prev,
                       next=next,
                       min_index=0,
                       max_index=200000,
                       shuffle=True,
                       step=step,
                       batch_size=batch_size)

#a, b = train_data.__next__()
#print(a, b)

valid_data = generator(weather_data,
                       prev=prev,
                       next=next,
                       min_index=200001,
                       max_index=210000,
                       step=step,
                       batch_size=batch_size)

test_data = generator(weather_data,
                      prev=prev,
                      next=next,
                      min_index=300001,
                      max_index=None,
                      step=step,
                      batch_size=batch_size)

# how many steps to draw from entire validation data, not including the last one( since 300000 - 200001 )
valid_steps = 210000 - 200001 - prev
test_steps = len(weather_data) - 300001 - prev

N_epochs = 2
model = Sequential()
#model.add(layers.Flatten(input_shape=(prev // step, weather_data.shape[-1])))
#model.add(layers.Dense(32, activation='relu'))
model.add(layers.GRU(32, input_shape=(None, weather_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')

# FIXME: cannot finish the training soon.
#        validation data is too much to train in a loop soon.
#        We can skip validation data without acc data in history.
H = model.fit_generator(train_data,
                        steps_per_epoch=500,
                        epochs=N_epochs,
                        validation_data=valid_data,
                        validation_steps=valid_steps)

#hp = HistoryPlot(N_epochs)
#hp.show(H)
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N_epochs), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N_epochs), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N_epochs), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, N_epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

