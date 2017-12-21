from keras import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense

from helpers import *

# loading our wordVectors
wordVectors = np.load('../../data/our_trained_wordvectors/wordvecs_sg_6.npy')
print('Loaded the word vectors!')

# loading our ids matrix
ids = np.load('../../data/our_trained_wordvectors/ids_sg_6.npy')

# splitting our data in train and test sets
x_train, x_test, y_train, y_test = split_data(ids, 0.9)

print('Build model...')

# Here we define embedding parameters useful for the Embedding Layer
max_features = wordVectors.shape[0]
max_seq_length = ids.shape[1]
embedding_size = wordVectors.shape[1]

# here we define the parameters for the convolutional Layer
# kernel sizes contains the size of the two windows used for grouping words to compute a new feature
filters_shapes = [2, 4]

# the input shape of the input for the convolutional layer
# that is a matrix with rows = number of words in the tweet
# and columns = number of the dimensions of the word vector
input_shape = (max_seq_length, embedding_size)

# filters used in convolutional layer
number_of_filters = 128

# here we define the parameters for the LSTM layer
lstm_output_size = 256

# here we define parameters for the training
batch_size = 100
epochs = 5

# creating the model structure
model = Sequential()

# First layer, embedding with the pre-trained word vectors, we train on these parameters
model.add(Embedding(max_features, embedding_size, input_length=max_seq_length, weights=[wordVectors],
                    trainable=True))

# Prevent overfitting
model.add(Dropout(0.2))

# creating the convolutional layer with different kernel sizes
conv_model = conv_different_kernels(number_of_filters, filters_shapes, max_sentence_length=max_seq_length,
                                    input_dim=input_shape)

model.add(conv_model)

# preventing overfitting
model.add(Dropout(0.2))

# Adding LSTM layer
model.add(LSTM(lstm_output_size))

# adding the final dense layer to compute the y_value
model.add(Dense(1, activation='sigmoid'))

# printing the structure of our model
model.summary()

# defining our optimizer
adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


# compiling our model
model.compile(loss='binary_crossentropy',
              optimizer=adam_optimizer,
              metrics=['accuracy'])

print('Train...')
print(y_train.shape)
print(x_train.shape)

# fitting the model with our train sets
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
with open("run_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("run_weights.h5")
print("Saved model to disk")
