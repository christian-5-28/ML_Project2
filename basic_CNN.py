from helpers import *

# loading the tweet datasets
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

# loading our dictionary
wordsList = np.load('skipgrams/word_list_sg_7.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
# wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8

# loading our wordVectors
wordVectors = np.load('skipgrams/wordvecs_sg_7.npy')

print('Loaded the word vectors!')

# here we loaded our ids matrix
ids = np.load('skipgrams/ids_sg_7.npy')

# here we split the ids matrix in train and test sets
x_train, x_test, y_train, y_test = split_data(ids, 0.9)

print('Build model...')

# Here we define embedding parameters useful for the Embedding Layer
max_features = 83782
max_seq_length = int(sum(numWords) / len(numWords)) + 5
embedding_size = 300  # first time


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
epochs = 10

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

model.add(Dropout(0.1))

# adding the dense layer
model.add(Dense(1, activation='sigmoid'))

# defining the optimizers
adam_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# adadelta_optimizer = keras.optimizers.Adadelta()

model.compile(loss='binary_crossentropy',
              optimizer=adam_optimizer,
              metrics=['accuracy'])

model.summary()

print('Train...')
print(y_train.shape)
print(x_train.shape)

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
with open("basic_cnn_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("basic_cnn_weights.h5")
print("Saved model to disk")







