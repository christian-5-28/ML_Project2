import numpy as np
import re
from random import randint
import tensorflow as tf
import matplotlib.pyplot as plt

# TODO: quote https://github.com/adeshpande3/LSTM-Sentiment-Analysis/blob/master/Oriole%20LSTM.ipynb


def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            # num = randint(1, len(positive_files))
            # Leaving 750*2 samples for testing
            num = randint(1, len(positive_files) - 750)
            labels.append([1, 0])
        else:
            # num = randint(len(positive_files)+1, len(positive_files)+len(negative_files))
            # Leaving 750*2 samples for testing
            num = randint(len(positive_files)+750, len(positive_files)+len(negative_files))
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels


def get_test_batch():
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

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")



from os import listdir
from os.path import isfile, join

path_positive = "/Users/lorenzotara/Documents/EPFL/Machine Learning/ML_Project2/twitter-datasets/train_pos.txt"
path_negative = "/Users/lorenzotara/Documents/EPFL/Machine Learning/ML_Project2/twitter-datasets/train_neg.txt"

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

# TODO: used to save mini files - wrong
# with open('mini_positive.txt', 'wb') as f:
#     np.savetxt(f, positive_files[:1000], fmt='%s')
#
# with open('mini_negative.txt', 'wb') as f:
#     np.savetxt(f, negative_files[:1000], fmt='%s')


num_files_total = len(numWords)
print('The total number of files is', num_files_total)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

# plt.hist(numWords, 50)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.axis([0, 40, 0, 25000])
# plt.show()

'''
From the histogram as well as the average number of words per file, we can safely say that most reviews will fall under 
20 words, which is the max sequence length value we will set.
'''

max_seq_length = 20

'''
we'll be using a much more manageable matrix that is trained using GloVe, a similar word vector generation model. 
The matrix  will contain 400,000 word vectors, each with a dimensionality of 50.
We're going to be importing two different data structures, one will be a Python list with the 400,000 words, and one 
will be a 400,000 x 50 dimensional embedding matrix that holds all of the word vector values.
'''

wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print('Loaded the word vectors!')

positive_files = positive_files_total
negative_files = negative_files_total
num_files_mini = len(positive_files) + len(negative_files)
'''
Now, let's convert to an ids matrix
'''
# ids = np.zeros((num_files_mini, max_seq_length), dtype='int32')
# file_counter = 0
# for line in positive_files:
#     index_counter = 0
#     cleaned_line = clean_sentences(line)  # Cleaning the sentence
#     split = cleaned_line.split()
#
#     for word in split:
#         try:
#             ids[file_counter][index_counter] = wordsList.index(word)
#         except ValueError:
#             ids[file_counter][index_counter] = 399999  # Vector for unkown words
#         index_counter = index_counter + 1
#
#         # If we have already seen maxSeqLength words, we break the loop of the words of a tweet
#         if index_counter >= max_seq_length:
#             break
#     file_counter = file_counter + 1
#
#     print("Steps to end: " + str(len(positive_files) + len(negative_files) - file_counter))
#
#
# for line in negative_files:
#     index_counter = 0
#     cleaned_line = clean_sentences(line)
#     split = cleaned_line.split()
#
#     for word in split:
#         try:
#             ids[file_counter][index_counter] = wordsList.index(word)
#         except ValueError:
#             ids[file_counter][index_counter] = 399999  # Vector for unkown words
#         index_counter = index_counter + 1
#
#         if index_counter >= max_seq_length:
#             break
#     file_counter = file_counter + 1
#
#     print("Steps to end: " + str(len(positive_files) + len(negative_files) - file_counter))
#
#
# np.save('ids_train_not_full.npy', ids)

ids = np.load('ids_train_not_full.npy')
print(ids.shape)

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
# TODO: try different LSTM Units!!! The bigger, the better, the slower
# TODO: find out num dimensions

batch_size = 24
lstm_units = 64
num_classes = 2
iterations = 50000
num_dimensions = 300  # Dimensions for each word vector # 300 lui mette 300, ma da dove gli esce fuori??

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

data = tf.Variable(tf.zeros([batch_size, max_seq_length, num_dimensions]), dtype=tf.float32)  # Is it useful?
# An embedding is a mapping from discrete objects, such as words, to vectors of real numbers.
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
lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

# TODO: add sequence_length - http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# Creates a recurrent neural network specified by RNNCell cell. data is the input
# (outputs) value contains the output of the RNN cell at every time instant. - https://stackoverflow.com/questions/44162432/analysis-of-the-output-from-tf-nn-dynamic-rnn-tensorflow-function
# _ it's the final state
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

# TODO: CHECK FOR
# another more advanced network architecture choice is to stack multiple LSTM cells on top of each other. This is where
# the final hidden state vector of the first LSTM feeds into the second. Stacking these cells is a great way to help the
# model retain more long term dependence information, but also introduces more parameters into the model, thus possibly
# increasing the training time, the need for additional training examples, and the chance of overfitting. For more
# information on how you can add stacked LSTMs to your model, check out Tensorflow's excellent documentation.
# https://www.tensorflow.org/tutorials/recurrent#stacking_multiple_lstms

'''
The first output of the dynamic RNN function can be thought of as the last hidden state vector. This vector will be 
reshaped and then multiplied by a final weight matrix and a bias term to obtain the final output values.
'''
# tf.truncated_normal: outputs random values from a truncated normal distribution.
weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
value = tf.transpose(value, [1, 0, 2])  # understand why the transpose and not value[:, -1, :] = last output of the cell
last = tf.gather(value, int(value.get_shape()[0]) - 1)  # https://www.tensorflow.org/versions/master/api_docs/python/tf/gather
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
For the optimizer, we’ll use Adam and the default learning rate of .001.
Using this learning rate after 70.000 iterations the loss starts going up.
First try: lr = 0.001
Second try: lr = 0.0001
'''
# TODO: remember to try different learning rates!
# TODO: try different optimizers!
# softmax_cross_entropy_with_logits: Measures the probability error in discrete classification tasks in which the
# classes are mutually exclusive (each entry is in exactly one class). For example, each CIFAR-10 image is labeled
# with one and only one label: an image can be a dog or a truck, but not both.
# reduce_mean: computes the mean of elements across dimensions of a tensor.

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

'''
If you’d like to use Tensorboard to visualize the loss and accuracy values, you can also run and 
the modify the following code.
'''
import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"


'''
TRAINING

The basic idea of the training loop is that we first define a Tensorflow session. Then, we load in a batch of reviews 
and their associated labels. Next, we call the session’s run function. This function has two arguments. The first is 
called the "fetches" argument. It defines the value we’re interested in computing. We want our optimizer to be computed 
since that is the component that minimizes our loss function. The second argument is where we input our feed_dict. This 
data structure is where we provide inputs to all of our placeholders. We need to feed our batch of reviews and our batch
of labels. This loop is then repeated for a set number of training iterations.
Instead of training the network in this notebook (which will take at least a couple of hours), we’ll load in a 
pretrained model.

While the following cell is running, use your terminal to enter the directory that contains this notebook, enter 
tensorboard --logdir=tensorboard, and visit http://localhost:6006/ with a browser to keep an eye on your training progress.

'''

# IT TAKES 40 MINUTES

sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):

    if i % 500 == 0:
        print('Iteration number: ', i)

    # Next Batch of reviews
    nextBatch, nextBatchLabels = get_train_batch()
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    # Write summary to Tensorboard
    if i % 50 == 0:
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)
        writer.flush()

    # Save the network every 10,000 training iterations
    if i % 10000 == 0 and i != 0:
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()


'''
TESTING
'''

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

iterations = 100
scores = []
for i in range(iterations):
    nextBatch, nextBatchLabels = get_test_batch()
    score = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
    scores.append(score)
    print("Accuracy for this batch:", (score) * 100)

print('Total accuracy:')
print(sum(scores) / len(scores))

'''
HYPERPARAMETER TUNING

Choosing the right values for your hyperparameters is a crucial part of training deep neural networks effectively. 
You'll find that your training loss curves can vary with your choice of optimizer (Adam, Adadelta, SGD, etc), learning 
rate, and network architecture. With RNNs and LSTMs in particular, some other important factors include the number of 
LSTM units and the size of the word vectors.
Learning Rate: RNNs are infamous for being difficult to train because of the large number of time steps they have. 
Learning rate becomes extremely important since we don't want our weight values to fluctuate wildly as a result of a 
large learning rate, nor do we want a slow training process due to a low learning rate. The default value of 0.001 is 
a good place to start. You should increase this value if the training loss is changing very slowly, and decrease if the 
loss is unstable.
Optimizer: There isn't a consensus choice among researchers, but Adam has been widely popular due to having the adaptive
learning rate property (Keep in mind that optimal learning rates can differ with the choice of optimizer).
Number of LSTM units: This value is largely dependent on the average length of your input texts. While a greater number 
of units provides more expressibility for the model and allows the model to store more information for longer texts, the
network will take longer to train and will be computationally expensive.
Word Vector Size: Dimensions for word vectors generally range from 50 to 300. A larger size means that the vector is 
able to encapsulate more information about the word, but you should also expect a more computationally expensive model.
'''
