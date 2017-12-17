import numpy as np
import re
from random import randint
import tensorflow as tf
from helpers import *
import datetime
import matplotlib.pyplot as plt

# TODO: quote https://github.com/adeshpande3/LSTM-Sentiment-Analysis/blob/master/Oriole%20LSTM.ipynb


def get_train_batch():
    """
    Returns a random batch from the train set and the corresponding labels
    """
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            # num = randint(1, len(positive_files))
            # Leaving 750*2 samples for testing
            num = randint(1, len(positive_files))
            labels.append([1, 0])
        else:
            # num = randint(len(positive_files)+1, len(positive_files)+len(negative_files))
            # Leaving 750*2 samples for testing
            num = randint(len(positive_files), len(positive_files)+len(negative_files))
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels


def get_test_batch():
    """
    Returns a random batch from the test set and the corresponding labels
    """
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        num = randint(len(positive_files) - 750, len(positive_files) + 750)
        if num <= len(positive_files):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels

'''
Loading the files
'''
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

print('Positive files finished')

with open(path_negative, "r", encoding='utf-8') as f:
    for line in f:
        negative_files_total.append(line)
        counter = len(line.split())
        numWords.append(counter)
print('Negative files finished')


# plt.hist(numWords, 50)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.axis([0, 40, 0, 25000])
# plt.show()

'''
Loading pre-trained wordvectors and wordsList
'''

wordVectors = np.load('skipgrams/wordvecs_sg_6.npy')

wordsList = np.load('skipgrams/word_list_sg_6.npy')
wordsList = wordsList.tolist()  # Originally loaded as numpy array

positive_files = positive_files_total
negative_files = negative_files_total
total_length = len(positive_files) + len(negative_files)

'''
Now, let's convert to an ids matrix
'''
# ids = create_ids_matrix(positive_files, negative_files, max_seq_length, wordsList)

ids = np.load('skipgrams/ids_sg_6.npy')
max_seq_length = ids.shape[1]

'''
Now, we’re ready to start creating our Tensorflow graph. We’ll first need to define some hyperparameters, such as batch 
size, number of LSTM units, number of output classes, and number of training iterations.

As with most Tensorflow graphs, we’ll now need to specify two placeholders, one for the inputs into the network, and 
one for the labels. The most important part about defining these placeholders is understanding each 
of their dimensionalities.
The labels placeholder represents a set of values, each either [1, 0] or [0, 1], depending on whether each training 
example is positive or negative. Each row in the integerized input placeholder represents the integerized representation
of each training example that we include in our batch.

Integerized Inputs - MAX SEQUENCE LENGTH (input_data)                       Labels - of classes (labels)
[    41    804 201534   1005     15   7446      5  13767      0      0]     [1,0]     -| 
[    41    804 201534   5005     15   7446      5  12767      0      0]     [1,0]      | 
[    10    820 201539   5005     35   4436      3  12767      0      0]     [0,1]      | 
.                                                                           .          |--- BATCH SIZE
.                                                                           .          | 
.                                                                           .          | 
[    13    333 222222   4342     12   5437      5  56456      0      0]     [1,0]     -| 
'''
batch_size = 100
lstm_units = 128
num_classes = 2
epochs = 2
# Dimensions for each word vector
num_dimensions = wordVectors.shape[1]

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, num_classes])  # a place in memory where we will store value later on.
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_length])

'''
Once we have our input data placeholder, we’re going to call the tf.nn.embedding_lookup() function in order to get our 
word vectors. The call to that function will return a 3-D Tensor of dimensionality batch size by max sequence length by 
word vector dimensions. In order to visualize this 3-D tensor, you can simply think of each data point in the 
integerized input tensor as the corresponding D dimensional vector that it refers to.
Dimension of data: batch size X max sequence length X word vector dimensions.
Number of tweets in the batch X max length of a tweet X dimensions of the word vector
'''

data = tf.Variable(tf.zeros([batch_size, max_seq_length, num_dimensions]), dtype=tf.float32)
# An embedding is a mapping from discrete objects, such as words to vectors of real numbers.
data = tf.nn.embedding_lookup(wordVectors, input_data)

'''
Now that we have the data in the format that we want, let’s look at how we can feed this input into an LSTM network. 
We’re going to call the tf.nn.rnn_cell.BasicLSTMCell function. This function takes in an integer for the number of LSTM
units that we want. This is one of the hyperparameters that will take some tuning to figure out the optimal value. We’ll 
then wrap that LSTM cell in a dropout layer to help prevent the network from overfitting.
Finally, we’ll feed both the LSTM cell and the 3-D tensor full of input data into a function called tf.nn.dynamic_rnn. 
This function is in charge of unrolling the whole network and creating a pathway for the data to flow through 
the RNN graph.
'''

lstmCell = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)
# We’ll then wrap that LSTM cell in a dropout layer to help prevent the network from overfitting
lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=0.9)

# Creates a recurrent neural network specified by RNNCell cell. data is the input
# (outputs) value contains the output of the RNN cell at every time instant. - https://stackoverflow.com/questions/44162432/analysis-of-the-output-from-tf-nn-dynamic-rnn-tensorflow-function
# _ it's the final state
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float64)

'''
The first output of the dynamic RNN function can be thought of as the last hidden state vector. This vector will be 
reshaped and then multiplied by a final weight matrix and a bias term to obtain the final output values.
'''
# tf.truncated_normal: outputs random values from a truncated normal distribution.
weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

# last output of the cell
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)

last = tf.cast(last, tf.float32)
prediction = (tf.matmul(last, weight) + bias)  # matrix product

'''
Next, we’ll define correct prediction and accuracy metrics to track how the network is doing. The correct prediction 
formulation works by looking at the index of the maximum value of the 2 output values, and then seeing 
whether it matches with the training labels.
'''
# argmax(input, axis=None, name=None, dimension=None, output_type=tf.int64)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# reduce_mean: computes the mean of elements across dimensions of a tensor.
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

'''
We’ll define a standard cross entropy loss with a softmax layer put on top of the final prediction values.
Best learning rate found: 0.001
'''

# softmax_cross_entropy_with_logits: Measures the probability error in discrete classification tasks in which the
# classes are mutually exclusive (each entry is in exactly one class). For example, each CIFAR-10 image is labeled
# with one and only one label: an image can be a dog or a truck, but not both.
# reduce_mean: computes the mean of elements across dimensions of a tensor.

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

'''
Visualizing the process
'''

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard_new/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"


'''
TRAINING - RIGUARDARE

While the following code is running, use your terminal to enter the directory that contains this notebook, enter 
tensorboard --logdir=tensorboard, and visit http://localhost:6006/ with a browser to keep an eye on your training progress.

'''

# IT TAKES 90 MINUTES PER EPOCH

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for epoch in range(epochs):

    x_tr, x_te, y_tr, y_te = split_data_tf(ids, 0.9)

    # for i in range(0, len(x_tr)-batch_size, batch_size):
    for i in range(0, len(x_tr)-batch_size, batch_size):

        if i/batch_size % 500 == 0:
            print('Iteration number: ', i/batch_size)
            print('Step to the end: ', (len(x_tr) - i)/batch_size)

        # Next Batch of reviews
        # nextBatch, nextBatchLabels = get_train_batch()
        nextBatch = x_tr[i:i+batch_size]
        nextBatchLabels = y_tr[i:i+batch_size]

        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if i/batch_size % 50 == 0:
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)
            writer.flush()

        # Save the network every 10,000 training iterations
        if i/batch_size % 100000 == 0 and i != 0:
            save_path = saver.save(sess, "models_new/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)

    predictions = []
    for i in range(0, len(y_te)-batch_size, batch_size):
        if i/batch_size % 10 == 0:
            print('Iteration number: ', i/batch_size, ' tot_iter = ', len(y_te)/batch_size)
        test_batch = x_te[i:i+batch_size]
        pred = sess.run([prediction], {input_data: test_batch})
        predictions += pred

    print("Epoch: ", epoch)

    final_predictions = []
    for elem in predictions:
        final_pred = np.argmax(elem)
        if final_pred == 0:
            final_predictions.append(1)
        else:
            final_predictions.append(-1)
    test_labels = []
    for elem in y_te:
        final_pred = np.argmax(elem)
        if final_pred == 0:
            test_labels.append(1)
        else:
            test_labels.append(-1)

    accuracy = np.array(final_predictions) == np.array(test_labels)
    acc = sum(accuracy)/len(accuracy)
    print("Accuracy: ", acc)

writer.close()


'''
PREDICT
'''
# path_test = "twitter-datasets/test_data.txt"
# test_files = []
#
# with open(path_test, "r") as f:
#     for line in f:
#         test_files.append(line)
#
# ids_test = np.zeros((len(test_files), max_seq_length), dtype='int32')
# file_counter = 0
# for line in test_files:
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
#     print("Steps to end (test): " + str(len(test_files) - file_counter))
#
#
#
# sess = tf.InteractiveSession()
# saver = tf.train.Saver()
# saver.restore(sess, tf.train.latest_checkpoint('models'))
#
# for i in range(0, len(test_files)):
#     pred = sess.run([prediction], {input_data: test_batch})
#
# make_submission(pred)