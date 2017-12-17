import pandas as pd
import datetime
import numpy as np
import re
import keras
from keras.models import Model, Input, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Embedding, Dropout, Conv1D, MaxPooling1D, Activation, \
    LSTM, BatchNormalization, Merge
from keras.utils import plot_model


def smooth_graph(y_value_list, smooth_window):

    smoothed_list = []

    for index, element in enumerate(y_value_list):

        window = min(index, smooth_window)

        temp_list = y_value_list[index - window : index + 1]

        mean_value = np.mean(temp_list)

        smoothed_list.append(mean_value)

    return smoothed_list


def create_word_list(documents, filter):
    '''
    Create word list of unique words which occurrencies are
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


def conv_different_kernels(num_filters, filters_shape, max_sentence_length, input_dim):

    parallel_convolutional_layers = []

    input_ = Input(shape=input_dim)

    for filter_shape in filters_shape:
        #layer = Conv2D(filters=num_filters, kernel_size=(filter_shape, 128), padding='same', activation='relu')(input_)
        layer = Conv1D(filters=num_filters, kernel_size=filter_shape, padding='same', activation='relu')(input_)
        #
        #layer = MaxPooling2D((1, 11), strides=(1, 1), padding='same')(layer)
        #layer = MaxPooling2D((max_sentence_length - filter_shape + 1, 128), strides=(1, 1), padding='same')(layer)
        layer = MaxPooling1D((max_sentence_length - filter_shape + 1), padding='same')(layer)

        parallel_convolutional_layers.append(layer)

    merged = keras.layers.concatenate(parallel_convolutional_layers, axis=1)
    #merged = Flatten()(merged)
    #out = Flatten()(merged)

    #out = Dense(200, activation='relu')(merged)
    #out = Dense(num_classes, activation='softmax')(out)

    model = Model(input_, outputs=merged)
    return model


def add_convolutional_block(current_model, num_filters, kernel_size, max_sentence_length):

    conv_layer = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same')
    current_model.add(conv_layer)

    current_model.add(BatchNormalization())

    current_model.add(Activation('sigmoid'))

    max_pool_layer = MaxPooling1D((max_sentence_length - kernel_size + 1), padding='same')

    current_model.add(max_pool_layer)

    return current_model


def add_conv_block_filter_sizes(num_filters, max_sentence_length, filter_shapes, input_dim):

    parallel_convolutional_layers = []

    input_ = Input(shape=input_dim)

    for filter_shape in filter_shapes:
        layer = Conv1D(filters=num_filters, kernel_size=filter_shape, padding='same')(input_)

        layer = BatchNormalization()(layer)

        layer = Activation('sigmoid')(layer)

        layer = MaxPooling1D((max_sentence_length - filter_shape + 1), padding='same')(layer)

        parallel_convolutional_layers.append(layer)

    merged = keras.layers.concatenate(parallel_convolutional_layers, axis=1)

    model = Model(input_, outputs=merged)

    return model


def clean_sentences(string):
    # Lowercases whole sentence, and then replaces the following
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


def remove_test_ids(path_test):
    test_files_total = []
    with open(path_test, "r") as f:
        for line in f:
            comma_index = line.index(',')
            line = line[comma_index + 1:]
            test_files_total.append(line)
    return np.savetxt('test_files_no_ids.txt', test_files_total, fmt="%s")


def combine_data():
    '''Combines the postive and negative files, maybe not needed in final version'''
    filenames = ['twitter-datasets/train_pos_full.txt', 'twitter-datasets/train_neg_full.txt', 'test_files_no_ids.txt']
    filenames2 = ['twitter-datasets/train_pos.txt', 'twitter-datasets/train_neg.txt', 'test_files_no_ids.txt']
    with open('twitter-datasets/combined_full.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    with open('twitter-datasets/combined.txt', 'w') as outfile:
        for fname in filenames2:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    return


def tokenize(s):
    '''Twitter customized tokenizer, NOT USED AT THE MOMENT'''
    # Heart symbol
    emoticons_str = r"""
        (?:
            [<] # heart top
            [3] # heart bottom
        )"""

    regex_str = [
        emoticons_str,
        r'<[^>]+>',  # HTML tags
        r'(?:@[\w_]+)',  # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
        r'(?:[\w_]+)',  # other words
        r'(?:\S)'  # anything else
    ]
    tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
    return tokens_re.findall(s)



