import numpy as np
import re
from helpers import *
from random import randint
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

max_seq_length = 20

wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print('Loaded the word vectors!')

path_test = "twitter-datasets/test_data.txt"
test_files = []

with open(path_test, "r") as f:
    for line in f:
        test_files.append(line)


# ids_test = np.zeros((len(test_files), max_seq_length), dtype='int32')
# indices = []
#
# file_counter = 0
# for line in test_files:
#     indices.append(line[0])
#     # print(line)
#     line = line[2:]
#     # print(line)
#     index_counter = 0
#     cleaned_line = clean_sentences(line)  # Cleaning the sentence
#     split = cleaned_line.split()
#
#     for word in split:
#         try:
#             ids_test[file_counter][index_counter] = wordsList.index(word)
#         except ValueError:
#             ids_test[file_counter][index_counter] = 399999  # Vector for unkown words
#         index_counter = index_counter + 1
#
#         # If we have already seen maxSeqLength words, we break the loop of the words of a tweet
#         if index_counter >= max_seq_length:
#             break
#     file_counter = file_counter + 1
#
#     if file_counter % 100 == 0:
#         print("Steps to end (test): " + str(len(test_files) - file_counter))
#
# np.save('ids_test.npy', ids_test)

ids_test = np.load('ids_test.npy')




batch_size = 50
lstm_units = 64
num_classes = 2
iterations = 100000
num_dimensions = 300  # Dimensions for each word vector # 300 lui mette 300, ma da dove gli esce fuori??

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_classes])  # a place in memory where we will store value later on.
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_length])

data = tf.Variable(tf.zeros([batch_size, max_seq_length, num_dimensions]), dtype=tf.float32)  # Is it useful?
# An embedding is a mapping from discrete objects, such as words, to vectors of real numbers.
data = tf.nn.embedding_lookup(wordVectors, input_data)


lstmCell = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)
# We’ll then wrap that LSTM cell in a dropout layer to help prevent the network from overfitting
lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

# Creates a recurrent neural network specified by RNNCell cell. data is the input
# (outputs) value contains the output of the RNN cell at every time instant. - https://stackoverflow.com/questions/44162432/analysis-of-the-output-from-tf-nn-dynamic-rnn-tensorflow-function
# _ it's the final state
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

# tf.truncated_normal: outputs random values from a truncated normal distribution.
weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
value = tf.transpose(value, [1, 0, 2])  # understand why the transpose and not value[:, -1, :] = last output of the cell
last = tf.gather(value, int(value.get_shape()[0]) - 1)  # https://www.tensorflow.org/versions/master/api_docs/python/tf/gather
prediction = (tf.matmul(last, weight) + bias)  # matrix product

# argmax(input, axis=None, name=None, dimension=None, output_type=tf.int64)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# reduce_mean: computes the mean of elements across dimensions of a tensor.
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))



sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

predictions = []
for i in range(0, len(test_files), batch_size):
    if i+batch_size < len(test_files):
        test_batch = ids_test[i:i+batch_size]
        pred = sess.run([prediction], {input_data: test_batch})

    else:
        test_batch = ids_test[i:len(test_files)]
        missed_words = i + batch_size - len(test_files)

        for i in range(missed_words):
            test_batch = np.vstack((test_batch, np.array([0]*max_seq_length)))

        pred = sess.run([prediction], {input_data: test_batch})

        pred[0] = pred[0][0:len(pred[0]) - missed_words]

    if i == 0:
        predictions = pred[0]
    else:
        predictions = np.vstack((predictions, pred[0]))

make_submission(predictions, 'first_total_prediction', from_tf=True)