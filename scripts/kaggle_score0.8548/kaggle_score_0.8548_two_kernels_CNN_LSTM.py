from keras import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense

from helpers import *


wordVectors = np.load('../../data/our_trained_wordvectors/wordvecs_sg_6.npy')
print('Loaded the word vectors!')

ids = np.load('../../data/our_trained_wordvectors/ids_sg_6.npy')

x_train, x_test, y_train, y_test = split_data(ids, 0.9)

print('Build model...')

# embedding parameters
max_features = wordVectors.shape[0]
max_seq_length = ids.shape[1]
embedding_size = wordVectors.shape[1]

# convolution parameters
filters_shapes = [2, 4]
input_shape = (max_seq_length, embedding_size)
number_of_filters = 128

# RNN parameters
lstm_output_size = 256

# Training
batch_size = 100
epochs = 5

# creating the model structure

model = Sequential()
# First layer, embedding
model.add(Embedding(max_features, embedding_size, input_length=max_seq_length, weights=[wordVectors],
                    trainable=False))  # w\o wordVectors - old one

# Prevent overfitting
model.add(Dropout(0.1))

model.summary()

# creating the parallel convolution layers
conv_model = conv_different_kernels(number_of_filters, filters_shapes, max_sentence_length=max_seq_length,
                                    input_dim=input_shape)

conv_model.summary()

model.add(conv_model)
model.add(Dropout(0.1))

# Adding LSTM layer
model.add(LSTM(lstm_output_size))


model.add(Dense(1, activation='sigmoid'))


model.summary()

adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


model.compile(loss='binary_crossentropy',
              optimizer=adam_optimizer,
              metrics=['accuracy'])

print('Train...')
print(y_train.shape)
print(x_train.shape)


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# serialize model to JSON
model_json = model.to_json()
with open("two_kernels_cnn_lstm_model.json", "w") as json_file:
    json_file.write(model_json)


# serialize weights to HDF5
model.save_weights("two_kernels_cnn_lstm_weights.h5")
print("Saved model to disk")







