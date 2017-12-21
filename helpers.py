import pandas as pd
import datetime
import numpy as np
import re
import keras
from keras.models import Model, Input, model_from_json
from keras.layers import Conv1D, MaxPooling1D


# UTILITIES FOR SPLITTING THE DATA

def split_data(x, ratio, seed=1):
    """split the dataset based on the split ratio."""

    y = np.array([1] * int(x.shape[0]/2))
    y = np.append(y, np.array([0] * int(x.shape[0]/2)))
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = x.shape[0]
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def split_data_tf(x, ratio, seed=1):
    """split the dataset based on the split ratio."""

    y = np.array([[1, 0]] * int(x.shape[0]/2))
    y = np.concatenate((y, np.array([[0, 1]] * int(x.shape[0]/2))))
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = x.shape[0]
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


# METHODS FOR THE PRE-PROCESSING OF OUR DATA

def clean_sentences(string):
    # Lower cases whole sentence, and then replaces the following

    string = string.lower().replace("<br />", " ")
    string = string.replace("n't", " not")
    string = string.replace("'m", " am")
    string = string.replace("'ll", " will")
    string = string.replace("'d", " would")
    string = string.replace("'ve", " have")
    string = string.replace("'re", " are")
    string = string.replace("'s", " is")
    string = string.replace("#", "<hashhtagg> ")
    string = string.replace("lol", "laugh")
    string = string.replace("<3", "love")
    string = string.replace("<user>", "")
    string = string.replace("<url>", "")

    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = re.sub(strip_special_chars, "", string.lower())

    # Tokenizes string:
    string = string.split()
    # string = tokenize(string)

    # Replaces numbers with the keyword <number>
    # string = [re.sub(r'\d+[.]?[\d*]?$', '<number>', w) for w in string]

    # Won't = will not, shan't = shall not, can't = can not
    string = [w.replace("wo", "will") for w in string]
    string = [w.replace("ca", "can") for w in string]
    string = [w.replace("sha", "shall") for w in string]

    # Any token which expresses laughter is replaced with "laugh"
    # for i, word in enumerate(string):
    #     if "haha" in word:
    #         string[i] = word.replace(word, "laugh")

    return string


def create_word_list(documents, filter):
    '''
    Create word list of unique words which occurrences are higher than filter
    '''
    word_dict = {}
    i = 0
    for document in documents:
        if i % 10000 == 0:
            print('Step to the end: ', len(documents) - i)
        split = clean_sentences(document)
        for word in split:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
        i += 1

    word_set = set()
    for key, value in word_dict.items():
        if value >= filter:
            word_set.add(str.encode(key))

    np.save('words_list_tweets_final.npy', list(word_set))
    return list(word_set)


def create_ids_matrix(positive_files, negative_files, max_seq_length, wordsList):
    '''
    Convert to an ids matrix
    '''
    total_files_length = len(positive_files) + len(negative_files)
    ids = np.zeros((total_files_length, max_seq_length), dtype='int32')
    file_counter = 0
    start_time = datetime.datetime.now()

    for line in positive_files:
        index_counter = 0
        split = clean_sentences(line)  # Cleaning the sentence

        for word in split:
            try:
                ids[file_counter][index_counter] = wordsList.index(word)
            except ValueError:
                ids[file_counter][index_counter] = len(
                    wordsList) - 1  # Vector for unknown positive vectors, not used otherwise
            index_counter = index_counter + 1

            # If we have already seen maxSeqLength words, we break the loop of the words of a tweet
            if index_counter >= max_seq_length:
                break

        if file_counter % 10000 == 0:
            print("Steps to end: " + str(total_files_length - file_counter))
            print('Time of execution: ', datetime.datetime.now() - start_time)

        file_counter = file_counter + 1

    del positive_files

    for line in negative_files:
        index_counter = 0
        split = clean_sentences(line)

        for word in split:
            try:
                ids[file_counter][index_counter] = wordsList.index(word)
            except ValueError:
                ids[file_counter][index_counter] = len(
                    wordsList) - 1  # Vector for unknown negative vectors, not used otherwise
            index_counter = index_counter + 1

            if index_counter >= max_seq_length:
                break

        if file_counter % 10000 == 0:
            print("Steps to end: " + str(total_files_length - file_counter))
            print('Time of execution: ', datetime.datetime.now() - start_time)
        file_counter = file_counter + 1

    np.save('ids_sg_6.npy', ids)


# UTILITY FOR CREATING THE KAGGLE SUBMISSION

def keras_prediction(model_path, weights_path, ids_test_path, csv_file_name):
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")

    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    ids_test = np.load(ids_test_path)

    prediction = loaded_model.predict(ids_test, verbose=0)

    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = -1
    prediction = prediction.reshape(-1)
    print(prediction)
    make_submission(prediction, csv_file_name)


def make_submission(pred, filename, from_tf=False):

    indices = np.arange(len(pred))
    # df = pd.DataFrame(columns=['Id','Prediction'])
    df = pd.DataFrame()

    for elem, idx in zip(pred, indices):
        if idx % 100 == 0:
            print('Prediction number: ', idx)

        if from_tf:
            final_pred = np.argmax(elem)
            if final_pred == 0:
                final_pred = 1
            else:
                final_pred = -1

        else:
            final_pred = int(elem)

        df = df.append([[idx+1, final_pred]], ignore_index=True)

    df.columns = ['Id', 'Prediction']
    print(df.tail())

    df.to_csv(filename, index=False)


# UTILITY METHOD FOR CREATION OF A CONVOLUTIONAL LAYER IN KERAS
def conv_different_kernels(num_filters, kernel_sizes, max_sentence_length, input_dim):
    """
    creates a convolutional layer with filters using different
    kernel sizes (window of words selected in order to compute
    a new feature value). It uses the Keras functional API in order
    to create the filters with different kernel size and then it
    merges the blocks created

    :param num_filters:
    :param kernel_sizes: list of the values of the kernel sizes
    :param max_sentence_length: length of the sentence
    :param input_dim: dimension of the input
    :return:
    """

    convolutional_layers = []

    input_ = Input(shape=input_dim)

    for kernel_size in kernel_sizes:
        layer = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu')(input_)

        layer = MaxPooling1D((max_sentence_length - kernel_size + 1), padding='same')(layer)

        convolutional_layers.append(layer)

    if len(convolutional_layers) > 1:
        merged = keras.layers.concatenate(convolutional_layers, axis=1)

    else:
        merged = convolutional_layers[0]

    model = Model(input_, outputs=merged)
    return model


# CLASS AND METHODS TO SAVE METRICS FROM OUR MODELS
class History(keras.callbacks.Callback):
    """
    we create a custom keras Callback in order to
    save for each batch the train loss and the
    train accuracy and for each epoch the validation
    loss and the validation accuracy
    """

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.epocs_losses = []
        self.epocs_acc = []
        self.epocs_val_loss = []
        self.epocs_val_acc = []


    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.epocs_val_loss.append(logs.get('val_loss'))
        self.epocs_val_acc.append(logs.get('val_acc'))


def smooth_graph(y_value_list, smooth_window):

    smoothed_list = []

    for index, element in enumerate(y_value_list):

        window = min(index, smooth_window)

        temp_list = y_value_list[index - window : index + 1]

        mean_value = np.mean(temp_list)

        smoothed_list.append(mean_value)

    return smoothed_list
