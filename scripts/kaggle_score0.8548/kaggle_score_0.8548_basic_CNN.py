from helpers import *

##################SCRIPT##################

path_positive = "twitter-datasets/train_pos_full.txt"
path_negative = "twitter-datasets/train_neg_full.txt"

numWords = []
positive_files_total = []
negative_files_total = []
with open(path_positive, "r") as f:
    for line in f:
        positive_files_total.append(line)
        counter = len(line.split())
        numWords.append(counter)

with open(path_negative, "r", encoding='utf-8') as f:
    for line in f:
        negative_files_total.append(line)
        counter = len(line.split())
        numWords.append(counter)

num_files_total = len(numWords)
print('The total number of files is', num_files_total)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords) / len(numWords))

wordsList = np.load('skipgrams/word_list_sg_7.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
# wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.load('skipgrams/wordvecs_sg_7.npy')

print('Loaded the word vectors!')

positive_files = positive_files_total
negative_files = negative_files_total
num_files_mini = len(positive_files) + len(negative_files)

ids = np.load('skipgrams/ids_sg_7.npy')

x_train, x_test, y_train, y_test = split_data(ids, 0.9)

print('Build model...')

# embedding parameters
max_features = 83782
max_seq_length = int(sum(numWords) / len(numWords)) + 5
# embedding_size = 64  # first time
embedding_size = 300  # first time
# embedding_size = 50
num_classes = 2

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

'''
# add second conv layer
model.add(Conv1D(number_of_filters,
                 3,
                 padding='same',
                 activation='relu',
                 strides=1))


filter_dim = 3
model.add(MaxPooling1D(6 - filter_dim + 1))

'''

# model.add(Flatten())
# model.add(Dense(1000, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# model.add(Dense(1))

# model.add(Activation('sigmoid'))

'''model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
'''

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
with open("basic_cnn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("basic_cnn_weights.h5")
print("Saved model to disk")

'''

model.add(Dropout(0.25))
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('tanh'))
'''






