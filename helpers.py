import pandas as pd
import numpy as np
import re
import datetime
import numpy as np
import re
from random import randint
import tensorflow as tf
import keras
from keras import Model, Input, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Embedding, Dropout, Conv1D, MaxPooling1D, Activation, \
    LSTM, BatchNormalization, Merge
from keras.utils import plot_model


def create_word_list(documents):

    word_set = set()
    i = 0
    for document in documents:
        if i % 10000 == 0:
            print('Step to the end: ', len(documents) - i)
        split = clean_sentences(document).split()
        for word in split:
            word_set.add(str.encode(word))

        i += 1

    np.save('words_list_tweets.npy', list(word_set))
    return list(word_set)


def create_word_list2(documents):

    word_dict = {}
    i = 0
    for document in documents:
        if i % 10000 == 0:
            print('Step to the end: ', len(documents) - i)
        split = clean_sentences_eigil(document)
        for word in split:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
        i += 1

    word_set = set()
    for key, value in word_dict.items():
        if value >= 15:
            word_set.add(str.encode(key))

    np.save('words_list_tweets_15.npy', list(word_set))
    return list(word_set)


def clean_sentences(string):
    # Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def create_ids_matrix(positive_files, negative_files, max_seq_length, wordsList):

    ids = np.zeros((len(positive_files) + len(negative_files), max_seq_length), dtype='int32')
    file_counter = 0
    start_time = datetime.datetime.now()
    for line in positive_files:
        index_counter = 0
        split = clean_sentences_eigil(line)  # Cleaning the sentence
        # split = cleaned_line.split()

        for word in split:
            try:
                ids[file_counter][index_counter] = wordsList.index(word)
            except ValueError:
                ids[file_counter][index_counter] = len(wordsList)  # Vector for unknown words
            index_counter = index_counter + 1

            # If we have already seen maxSeqLength words, we break the loop of the words of a tweet
            if index_counter >= max_seq_length:
                break

        if file_counter % 10000 == 0:
            print("Steps to end: " + str(len(positive_files) + len(negative_files) - file_counter))
            print('Time of execution: ', datetime.datetime.now() - start_time)

        file_counter = file_counter + 1

    for line in negative_files:
        index_counter = 0
        cleaned_line = clean_sentences(line)
        split = cleaned_line.split()

        for word in split:
            try:
                ids[file_counter][index_counter] = wordsList.index(word)
            except ValueError:
                ids[file_counter][index_counter] = len(wordsList)  # Vector for unkown words
            index_counter = index_counter + 1

            if index_counter >= max_seq_length:
                break

        if file_counter % 10000 == 0:
            print("Steps to end: " + str(len(positive_files) + len(negative_files) - file_counter))
            print('Time of execution: ', datetime.datetime.now() - start_time)
        file_counter = file_counter + 1

    np.save('ids_from_tweets.npy', ids)


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


def load_data():
    # TODO: 195. 804 elements instead of 200.000
    '''
    Loading the tweets
    '''
    pos_data = pd.read_table('twitter-datasets/train_pos.txt', header=None)

    pos_data.columns = ['tweet']
    pos_data['label'] = 1

    neg_data = pd.read_table('twitter-datasets/train_pos.txt', header=None)

    neg_data.columns = ['tweet']
    neg_data['label'] = -1

    data = pos_data.append(neg_data, ignore_index=True)

    return data


def load_lexicons_sense_level():
    '''
    Loading the lexicons - Wrong WROOONG
    '''
    # TODO: quote https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730

    lexicons = pd.read_table('lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Senselevel-v0.92.txt', sep='--|\t',
                             names=('term', 'syn', 'emotion', 'value'), engine='python')

    lexicons.syn = lexicons.syn.str.split(',')

    syn = lexicons.groupby('term').first()[['syn']]
    syn.reset_index(inplace=True)
    no_syn = lexicons.drop('syn', axis=1)
    terms = no_syn.term
    emotions = no_syn.pivot(columns='emotion', values='value')
    emotions['term'] = terms
    emotions.set_index('term')

    emotions.fillna(0, inplace=True)
    print(emotions.head(5))
    emotions = emotions.groupby('term').sum()
    emotions.reset_index(inplace=True)
    emotions = pd.merge(emotions, syn, on='term')

    return emotions


def load_lexicons():

    # TODO: quote https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730

    lexicons = pd.read_table('lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', sep='\t',
                             names=('term', 'emotion', 'value'))

    terms = lexicons.term

    emotions = lexicons.drop('term', axis=1).pivot(columns='emotion', values='value')
    emotions['term'] = terms
    emotions.fillna(0, inplace=True)
    emotions = emotions.groupby('term').sum()
    emotions.reset_index(inplace=True)

    return emotions


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

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())



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

def clean_sentences_eigil(string):
    # Lowercases whole sentence, and then replaces the following
    string = string.lower().replace("<br />", " ")
    string = string.replace("n't", " not")
    string = string.replace("'m", " am")
    string = string.replace("'ll", " will")
    string = string.replace("'d", " would")
    string = string.replace("'ve", " have")
    string = string.replace("'re", " are")
    string = string.replace("'s", " is")
    string = string.replace("#", "<hashtag> ")
    string = string.replace("lol", "laugh")
    string = string.replace("<3", "love")

    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    re.sub(strip_special_chars, "", string.lower())

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






