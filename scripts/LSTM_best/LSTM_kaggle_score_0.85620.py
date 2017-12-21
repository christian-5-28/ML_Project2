"""
LSTM network
"""

import tensorflow as tf
from helpers import *
import datetime


'''
Loading pre-trained wordvectors and wordsList
'''

wordVectors = np.load('../../data/our_trained_wordvectors/wordvecs_sg_6.npy')

'''
Now, let's load our ids matrix
'''
ids = np.load('../../data/our_trained_wordvectors/ids_sg_6.npy')
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
epochs = 4

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
lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=0.85)

# Creates a recurrent neural network specified by RNNCell cell. data is the input
# (outputs) value contains the output of the RNN cell at every time instant.
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
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"


'''
TRAINING

While the following code is running, use your terminal to enter the directory that contains this notebook, enter 
tensorboard --logdir=scripts/LSTM_best/tensorboard, and visit http://localhost:6006/ with a browser to keep an eye on your training progress.

'''

# IT TAKES 100 MINUTES PER EPOCH

# NEW SESSION
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# RESTORE SESSION
# sess = tf.InteractiveSession()
# writer = tf.summary.FileWriter(logdir, sess.graph)
# saver = tf.train.Saver()
# saver.restore(sess, tf.train.latest_checkpoint('models_last'))


for epoch in range(0, epochs):

    # Uncomment only if you want to see different epochs in different colors on tensorBoard

    # if epoch > 1:
    #     save_path = saver.save(sess, "models_new/pretrained_lstm.ckpt", global_step=final_save)
    #     print("saved to %s" % save_path)
    #     writer.close()
    #     sess.close()
    #     sess = tf.InteractiveSession()
    #     writer = tf.summary.FileWriter(logdir, sess.graph)
    #     saver = tf.train.Saver()
    #     saver.restore(sess, tf.train.latest_checkpoint('models_new'))

    x_tr, x_te, y_tr, y_te = split_data_tf(ids, 0.9)

    for i in range((len(x_tr)-batch_size) * epoch, (len(x_tr)-batch_size)*(epoch+1), batch_size):

        index = i - (len(x_tr)-batch_size) * epoch

        if i/batch_size % 500 == 0:
            print('Iteration number: ', i/batch_size * (epoch+1))
            print('Step to the end: ', (len(x_tr) - i*(epoch+1))/batch_size)

        # Next Batch of reviews
        nextBatch = x_tr[index:index+batch_size]
        nextBatchLabels = y_tr[index:index+batch_size]

        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if i/batch_size % 50 == 0:
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)
            writer.flush()

        # Save the network every 10,000 training iterations
        if i % 10000 == 0 and i != 0:
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)

        final_save = i

    # Calculating the accuracy on the test set
    predictions = []
    for t in range(0, len(y_te), batch_size):
        if t/batch_size % 10 == 0:
            print('Iteration number: ', t/batch_size, ' tot_iter = ', len(y_te)/batch_size)
        test_batch = x_te[t:t+batch_size]
        pred = sess.run([prediction], {input_data: test_batch})
        if t == 0:
            predictions = pred[0]
        else:
            predictions = np.vstack((predictions, pred[0]))

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

    print(len(final_predictions))
    print(len(test_labels))

    accuracy = np.array(final_predictions) == np.array(test_labels)
    acc = sum(accuracy)/len(accuracy)
    print("Accuracy: ", acc)

save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=final_save)
print("saved to %s" % save_path)
writer.close()
