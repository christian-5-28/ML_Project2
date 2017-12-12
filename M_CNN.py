from keras.layers import ZeroPadding1D

from ML_Project2.helpers import *

##################SCRIPT##################

path_positive = "data/twitter-datasets/train_pos.txt"
path_negative = "data/twitter-datasets/train_neg.txt"

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
print('The average number of words in the files is', sum(numWords)/len(numWords))


wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print('Loaded the word vectors!')

positive_files = positive_files_total
negative_files = negative_files_total
num_files_mini = len(positive_files) + len(negative_files)

ids = np.load('ids_train_not_full.npy')

x_train, x_test, y_train, y_test = split_data(ids, 0.8)

print('Build model...')

# embedding parameters
max_features = 400000
max_seq_length = int(sum(numWords)/len(numWords)) + 5
# embedding_size = 64  # first time
embedding_size = 50  # first time
# embedding_size = 50
num_classes = 2

# Training
batch_size = 50
epochs = 2


# convolution parameters

nb_filter = 200
filter_length = 6
hidden_dims = nb_filter

embedding_matrix = np.load("ids_train_not_full.npy")
max_features = 400000
embedding_dims = 50

model = Sequential()

# main_input = Input(batch_shape=(None, max_seq_length), dtype='int32', name='main_input')

model.add(Embedding(max_features, embedding_dims, init='lecun_uniform', input_length=max_seq_length))

model.add(ZeroPadding1D(filter_length - 1))

model.add(Conv1D(nb_filter=nb_filter,
                 filter_length=filter_length,
                 border_mode='valid',
                 activation='relu',
                 subsample_length=1))

model.add(MaxPooling1D(pool_length=4, stride=2))

conv2 = Conv1D(nb_filter=nb_filter,
                 filter_length=filter_length,
                 border_mode='valid',
                 activation='relu',
                 subsample_length=1)

model.add(conv2)

output_shape = model.output_shape

model.add(MaxPooling1D(pool_length=output_shape[1]))

model.add(Flatten())

model.add(Dense(hidden_dims))

model.add(Dense(1, activation='softmax', init='lecun_uniform'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# serialize model to JSON
model_json = model.to_json()
with open("M_CNN_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("M_CNN_weights.h5")
print("Saved model to disk")