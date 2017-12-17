'''
Train a recurrent convolutional network.
Layers: embedding, Conv1D, LSTM
Final score on Kaggle: 0.8526
'''
from helpers import *
import numpy as np
import matplotlib.pyplot as plt


class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.epocs_losses = []
        self.epocs_acc = []
        self.epocs_val_loss = []
        self.epocs_val_acc = []

    def on_epoch_begin(self, epoch, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.epocs_losses.append(self.losses)
        self.epocs_acc.append(self.accuracy)
        self.epocs_val_loss.append(logs.get('val_loss'))
        self.epocs_val_acc.append(logs.get('val_acc'))


wordVectors = np.load('skipgrams/wordvecs_sg_6.npy')

wordsList = np.load('skipgrams/word_list_sg_6.npy')
wordsList = wordsList.tolist()  # Originally loaded as numpy array

ids = np.load('skipgrams/ids_sg_6.npy')
max_seq_length = ids.shape[1]

print(wordVectors.shape)


x_train, x_test, y_train, y_test = split_data(ids, 0.9)

print('Build model...')

# Embedding
# number of words in the word vectors
max_features = wordVectors.shape[0]
embedding_size = wordVectors.shape[1]

# Convolution
kernel_size = 5
filters = 128
pool_size = 16

# LSTM
lstm_output_size = 256

# Training
batch_size = 100
epochs = 5

drop_out = 0.1


model = Sequential()
# First layer, embedding using pretrained wordvectors
model.add(Embedding(max_features, embedding_size, weights=[wordVectors], input_length=max_seq_length, trainable=False))
# Prevent overfitting
model.add(Dropout(drop_out))
# First real layer, convolutional layer
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 use_bias=False))
# TODO: ask Christian
model.add(MaxPooling1D(pool_size=pool_size))
# Prevent overfitting
model.add(Dropout(drop_out))
# Adding LSTM layer
model.add(LSTM(lstm_output_size))
# Adding last layer, Dense(1)
model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
history = History()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=[history])

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# serialize model to JSON
model_json = model.to_json()
with open("crnn5_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("crnn5_weights.h5")
print("Saved model to disk")

print(history.epocs_val_acc)
print(len(history.epocs_val_acc))
print(history.epocs_val_loss)
print(len(history.epocs_val_loss))

fig = plt.figure(1)
plt.plot(history.accuracy)
plt.show()
fig.savefig('crnn_accuracy.png')
plt.close()

fig = plt.figure(2)
plt.plot(history.losses)
plt.show()
fig.savefig('crnn_losses.png')
plt.close()

# accuracy
# [0.8422999982833862, 0.84824799892902369, 0.84934399983882902, 0.85181999943256381, 0.85254799904823308]

# loss
# [0.34534629065394401, 0.33648159590959548, 0.33353952446579932, 0.32893382251262665, 0.32868177694678308]
