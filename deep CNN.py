from ML_Project2.helpers import *

##################SCRIPT##################

path_positive = "data/twitter-datasets/train_pos_full.txt"
path_negative = "data/twitter-datasets/train_neg_full.txt"

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

wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print('Loaded the word vectors!')

positive_files = positive_files_total
negative_files = negative_files_total
num_files_mini = len(positive_files) + len(negative_files)

ids = np.load('ids_final.npy')

x_train, x_test, y_train, y_test = split_data(ids, 0.8)

print('Build model...')

# embedding parameters
max_features = 400000
max_seq_length = int(sum(numWords) / len(numWords)) + 5
# embedding_size = 64  # first time
embedding_size = 128  # first time
# embedding_size = 50
num_classes = 2

# convolution parameters

filters_shapes = [2, 4, 5]
input_shape = (max_seq_length, embedding_size)
number_of_filters = 100

# RNN parameters
lstm_output_size = 100

# Training
batch_size = 50
epochs = 2

# creating the model structure

model = Sequential()
# First layer, embedding
model.add(Embedding(max_features, embedding_size, input_length=max_seq_length))  # w\o wordVectors - old one

# Prevent overfitting
# model.add(Dropout(0.25))

for index in range(0, 3):
    model = add_convolutional_block(model, pow(2, index)*128, 3, max_seq_length)

#model.add(Dropout(0.25))

# Adding LSTM layer
model.add(LSTM(lstm_output_size))

# Prevent overfitting
model.add(Dropout(0.25))

# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

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
with open("crnn2_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("crnn2_weights.h5")
print("Saved model to disk")

