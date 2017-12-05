import numpy as np
import re
from random import randint
import tensorflow as tf
import keras
from keras import Model, Input, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Embedding, Dropout, Conv1D, MaxPooling1D, Activation, \
    LSTM, BatchNormalization, Merge
from keras.utils import plot_model


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






