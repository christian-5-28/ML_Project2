from helpers import *


# loading our wordVectors
wordVectors = np.load('skipgrams/wordvecs_sg_6.npy')
print('Loaded the word vectors!')

# here we loaded our ids matrix
ids = np.load('skipgrams/ids_sg_6.npy')

# here we split the ids matrix in train and test sets
x_train, x_test, y_train, y_test = split_data(ids, 0.9)

print('Build model...')

# Here we define embedding parameters useful for the Embedding Layer
max_features = wordVectors.shape[0]
max_seq_length = ids.shape[1]
embedding_size = wordVectors.shape[1]


# here we define the parameters for the convolutional Layer
# kernel sizes contains the size of the two windows used for grouping words to compute a new feature
kernel_sizes = [2, 4]

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

# First layer, embedding with the pre-trained word vectors, we do not train these again
model.add(Embedding(max_features, embedding_size, input_length=max_seq_length, weights=[wordVectors],
                    trainable=False))  # w\o wordVectors - old one

# Prevent overfitting
model.add(Dropout(0.1))


# creating the convolutional layer with different kernel sizes
conv_model = conv_different_kernels(number_of_filters, kernel_sizes, max_sentence_length=max_seq_length,
                                    input_dim=input_shape)

model.add(conv_model)

model.add(Dropout(0.1))

# Adding LSTM layer
model.add(LSTM(lstm_output_size))

# adding the dense layer for reshaping and evaluate the y-value
model.add(Dense(1, activation='sigmoid'))

# defining the optimizers. We used the default values
# as suggested on the Keras documentation, we explicitly reported
# them for clarity.
adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# compiling our model structure
model.compile(loss='binary_crossentropy',
              optimizer=adam_optimizer,
              metrics=['accuracy'])

# here we print the structure of our model
model.summary()

print('Train...')
print(y_train.shape)
print(x_train.shape)

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
with open("basic_cnn_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("basic_cnn_weights.h5")
print("Saved model to disk")

# saving the validation accuracy for each epoch
val_acc_epochs = history.epocs_val_acc
np.save("val_acc_basic_CNN.npy", val_acc_epochs)

# saving the loss accuracy for each epoch
val_loss_epochs = history.epocs_val_loss
np.save("val_loss_basic_CNN.npy", val_loss_epochs)

# here we use our utility function "smooth_graph"
# in order to have smoothed metrics for the plot
smoothed_accuracy = smooth_graph(history.accuracy, 100)

# saving the smoothed metrics
np.save("smoothed_acc_CNN_LSTM.npy", smoothed_accuracy)

# same for the train loss metrics
smoothed_losses = smooth_graph(history.losses, 100)
np.save("smoothed_loss_CNN_LSTM.npy", smoothed_losses)





