"""
kaggle_score_0.8526_one_kernel_CNN_LSTM.py: this script creates a single-kernel-sized
CNN_LSTM architecture that obtained a score on Kaggle of 0.8526. In this script
we used our word vectors and ids matrix created in the pre-processing part.
Tested on macOs machine with a computing time of 100 minutes.
"""

__author__    = "Christian Sciuto, Eigil Lippert and Lorenzo Tarantino"
__copyright__ = "Copyright 2017, Second Machine Learning Project, EPFL Machine Learning Course CS-433, Fall 2017"
__credits__   = ["Christian Sciuto", "Eigil Lippert", "Lorenzo Tarantino"]
__license__   = "MIT"
__version_    = "1.0.1"
__status__    = "Project"

from keras import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense, Activation

from helpers import *
import numpy as np


# loading our word vectors
wordVectors = np.load('../../data/our_trained_wordvectors/wordvecs_sg_6.npy')
print(wordVectors.shape)
print('Loaded the word vectors!')

# ids = create_ids_matrix(positive_files, negative_files, max_seq_length, wordsList
ids = np.load('../../data/our_trained_wordvectors/ids_sg_6.npy')

x_train, x_test, y_train, y_test = split_data(ids, 0.9)

print('Build model...')

# Embedding
# number of words in the word vectors
max_features = wordVectors.shape[0]
embedding_size = wordVectors.shape[1]
max_seq_length = ids.shape[1]

# Convolution
kernel_size = 5
filters = 128
pool_size = 16

# LSTM
lstm_output_size = 256

drop_out = 0.1

# Training
batch_size = 100
epochs = 5

model = Sequential()

# First layer, embedding using pretrained wordvectors
model.add(Embedding(max_features, embedding_size, weights=[wordVectors], input_length=max_seq_length, trainable=False))

# Prevent overfitting
model.add(Dropout(drop_out))

# convolutional layer
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 use_bias=False))

# adding the max_pooling layer to decrease the number
# of features created after the convolutional layer
model.add(MaxPooling1D(pool_size=pool_size))

# Prevent overfitting
model.add(Dropout(drop_out))

# Adding LSTM layer
model.add(LSTM(lstm_output_size))

# adding the dense layer for reshaping and evaluate the y-value
model.add(Dense(1))
model.add(Activation('sigmoid'))

# compiling our model structure
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')

# defining our callback to save metrics
# in order to create the plots (loss, accuracy)
history = History()

# fitting the model with our data
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# evaluating our model on the test sets
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# serialize model to JSON
model_json = model.to_json()
with open("one_kernel_CNN_LSTM_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("one_kernel_CNN_LSTM_weights.h5")
print("Saved model to disk")

# creating the prediction on test set csv file
keras_prediction(model_path="one_kernel_CNN_LSTM_model.json",
                 weights_path="one_kernel_CNN_LSTM_weights.h5",
                 ids_test_path="../../data/our_trained_wordvectors/ids_test_sg_6.npy",
                 csv_file_name="one_kernel_cnn_lstm_prediction.csv")

# From here, we save our metrics results for the comparison with plots
val_acc_epochs = history.epocs_val_acc
np.save("val_acc_one_kernel_CNN_LSTM.npy", val_acc_epochs)

val_loss_epochs = history.epocs_val_loss
np.save("val_loss_one_kernel_CNN_LSTM.npy", val_loss_epochs)

smoothed_accuracy = smooth_graph(history.accuracy, 100)
np.save("smoothed_acc_one_kernel_CNN_LSTM.npy", smoothed_accuracy)

smoothed_losses = smooth_graph(history.losses, 100)
np.save("smoothed_loss_one_kernel_CNN_LSTM.npy", smoothed_losses)

