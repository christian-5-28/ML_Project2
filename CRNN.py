'''
Train a recurrent convolutional network.
Layers: embedding, Conv1D, LSTM
Final score on Kaggle: 0.8526
'''
from helpers import *
import numpy as np

# loading our wordVectors
wordVectors = np.load('skipgrams/wordvecs_sg_6.npy')

# here we loaded our ids matrix
ids = np.load('skipgrams/ids_sg_7.npy')

# here we split the ids matrix in train and test sets
x_train, x_test, y_train, y_test = split_data(ids, 0.9)

print('Build model...')

# Embedding
# number of words in the word vectors
max_seq_length = ids.shape[1]
max_features = wordVectors.shape[0]
embedding_size = wordVectors.shape[1]

# Convolution parameters
kernel_size = 5
filters = 128
pool_size = 16

# LSTM parameters
lstm_output_size = 256

# Training parameters
batch_size = 100
epochs = 5

# dropout value
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

# adding the max_pooling layer to decrease the number of features
# created after the convolutional layer
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

# defining our callback to save metrics in order to create the plots (loss, accuracy)
history = History()

# fitting the model with our data
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=[history])

# evaluating our model on the test sets
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

# From here, we save our metrics results for the comparison with plots

val_acc_epochs = history.epocs_val_acc
np.save("val_acc_CRNN.npy", val_acc_epochs)

val_loss_epochs = history.epocs_val_loss
np.save("val_loss_CRNN.npy", val_loss_epochs)

smoothed_accuracy = smooth_graph(history.accuracy, 100)
np.save("smoothed_acc_CRNN.npy", smoothed_accuracy)

smoothed_losses = smooth_graph(history.losses, 100)
np.save("smoothed_loss_CRNN.npy", smoothed_losses)

