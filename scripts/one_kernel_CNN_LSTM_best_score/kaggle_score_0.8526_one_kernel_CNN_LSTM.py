'''
Train a recurrent convolutional network.
Layers: embedding, Conv1D, LSTM
Final score on Kaggle: 0.8526
'''
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
# Ask Christian
model.add(MaxPooling1D(pool_size=pool_size))
# Prevent overfitting
model.add(Dropout(drop_out))
# Adding LSTM layer
model.add(LSTM(lstm_output_size))
# Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise
# activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a
# bias vector created by the layer (only applicable if use_bias is True).
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''
metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use 
metrics=['accuracy']. To specify different metrics for different outputs of a multi-output model, you could also pass a 
dictionary, such as metrics={'output_a': 'accuracy'}.
'''
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
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
