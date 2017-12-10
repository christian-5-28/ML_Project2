from gensim import corpora, models, similarities
import collections
import numpy as np
import os
import re
import logging
from keras import layers, preprocessing
from keras import models as mods
import random


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
    string = string.replace("#", "<hashtag> ")
    string = string.replace("lol", "laugh")

    # Tokenizes string:
    string = string.split()

    # Replaces <3 symbols
    string = [w.replace("<3", "love") for w in string]

    # Replaces numbers with the keyword <number>
    string = [re.sub(r'\d+[.]?[\d*]?$', '<number>', w) for w in string]

    # Won't = will not, shan't = shall not
    string = [w.replace("wo", "will") for w in string]
    string = [w.replace("sha", "shall") for w in string]

    # Any token which expresses laughter is replaced with "laugh"
    for i, word in enumerate(string):
        if "haha" in word or re.match(r'^haha', word) or re.match(r'^ahaha', word):
            string[i] = word.replace(word, "laugh")

    return string

def comine_pos_neg():
    '''Combines the postive and negative files, maybe not needed in final version'''
    filenames = ['twitter-datasets/train_pos_full.txt', 'twitter-datasets/train_neg_full.txt']
    filenames2 = ['twitter-datasets/train_pos.txt', 'twitter-datasets/train_neg.txt']
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


def read_data(path_comb):
    """Extract the first tweets enclosed as a list of words."""
    all_files_total = []
    with open(path_comb, "r") as f:
        for line in f:
            all_files_total.append(line)

    vocabulary = list()
    for line in all_files_total[:1000]:
        cleaned_line = clean_sentences(line)  # Cleaning the sentence
        vocabulary.extend(cleaned_line)  # Creating a list of all words in our tweets
    return vocabulary


def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


def get_sim(valid_word_idx, vocab_size):
    sim = np.zeros((vocab_size,))
    in_arr1 = np.zeros((1,))
    in_arr2 = np.zeros((1,))
    for i in range(vocab_size):
        in_arr1[0,] = valid_word_idx
        in_arr2[0,] = i
        out = validation_model.predict_on_batch([in_arr1, in_arr2])
        sim[i] = out
    return sim


class MyCorpus(object):
    def __iter__(self):
        for line in open(path_positive, encoding="utf-8"):
            # assume there's one document per line, tokens separated by whitespace
            line = clean_sentences(line)
            yield line


'''Iterator object that iterates through files in directory, picking every sentence from the file.
This is why "combined_full.txt" should be placed in its own directory.'''
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield clean_sentences(line)


# Based on:
# TODO: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
# TODO: http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
# and
# TODO: https://rare-technologies.com/word2vec-tutorial/

path_positive = "twitter-datasets/train_pos.txt"
path_negative = "twitter-datasets/train_neg.txt"
# path_combined = "twitter-datasets/combined_full.txt"

'''Loading senctences in a memory-friendly way, needs full path'''
sentences = MySentences("/Users/eyu/Google Drev/DTU/5_semester/ML/ML_Project2/combined_tweets")  # a memory-friendly iterator
# sentences = MyCorpus()
vector_dim = 300  # dimensions of word vectors


'''For logging the process'''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''Gensim model computation, either load existing or compute from scratch'''
# TODO: for info on params check, https://radimrehurek.com/gensim/models/word2vec.html
# model = models.Word2Vec.load("/Users/eyu/Google Drev/DTU/5_semester/ML/ML_Project2/gensim models/model1")
model = models.word2vec.Word2Vec(sentences, sg=1, iter=10, min_count=15, size=300, workers=3,
                                 negative=5)

'''Some possibilities for the wv object'''
# # A word vector for some word, can be accessed like this:
# print(model.wv['the'])

# get the most common words
print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])

# get the least common words
vocab_size = len(model.wv.vocab)
print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3])

'''Extracting vocabulary and the indexes'''
# convert the input data into a list of integer indexes aligning with the wv indexes
str_data = read_data(path_combined)
index_data = convert_data_to_index(str_data, model.wv)
print(str_data[:4], index_data[:4])


'''Saving and converting'''
# saves the model
model.save("/Users/eyu/Google Drev/DTU/5_semester/ML/ML_Project2/gensim models/model2")

# convert the wv word vectors into a numpy matrix that is suitable for insertion
# into our TensorFlow and Keras models
embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

thefile = open('vocab.txt', 'w')
for item in str_data:
    thefile.write("%s\n" % item)

thefile1 = open('vocab_indeces.txt', 'w')
for item in index_data:
    thefile1.write("%s\n" % item)

np.savetxt('wordvecs.txt', embedding_matrix)


'''Validation of the similiar words with Keras (for qualitative analysis)'''
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# input words - in this case we do sample by sample evaluations of the similarity
valid_word = layers.Input((1,), dtype='int32')
other_word = layers.Input((1,), dtype='int32')
# setup the embedding layer
embeddings = layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix])
embedded_a = embeddings(valid_word)
embedded_b = embeddings(other_word)
similarity = layers.merge([embedded_a, embedded_b], mode='cos', dot_axes=2)
# create the Keras model
validation_model = mods.Model(input=[valid_word, other_word], output=similarity)

# now run the model and get the closest words to the valid examples
for i in range(valid_size):
    valid_word = model.wv.index2word[valid_examples[i]]
    top_k = 8  # number of nearest neighbors
    sim = get_sim(valid_examples[i], len(model.wv.vocab))
    nearest = (-sim).argsort()[1:top_k + 1]
    log_str = 'Nearest to %s:' % valid_word
    for k in range(top_k):
        close_word = model.wv.index2word[nearest[k]]
        log_str = '%s %s,' % (log_str, close_word)
    print(log_str)




    # Nearest to u: shorty, yu, urs, shld, you, bak, ye, mayb,
# Nearest to my: chem, charlie, ma, everrr, panda, your, mah, wittle,
# Nearest to me: meh, mee, bak, cell, creep, steph, sean, us,
# Nearest to am: im, iam, dealer, eee, youre, personally, wayyy, messaged,
# Nearest to how: ey, missy, nahhh, jeff, sammy, eee, toy, where,
# Nearest to in: visiting, locked, sf, across, near, professor, lit, burning,
# Nearest to ': kairo, happenin, vas, rangers, yeaah, prince, twinkle, illegal,
# Nearest to im: am, youre, nicee, iam, av, cba, hes, sry,
# Nearest to do: nor, offended, exist, dig, spoil, explain, pretend, yepp,
# Nearest to (: ), cont, marie, viewers, georgia, 178cm, cr, 175cm,
# Nearest to like: familiar, mature, terrible, arab, split, tat, freaky, thy,
# Nearest to of: frees, marks, receiving, mcfly, emotions, including, nation, seven,
# Nearest to is: isnt, sees, plays, absolute, iss, treats, hates, uses,
# Nearest to good: goood, great, refreshing, mutual, nicee, tiring, lush, fab,
# Nearest to that: proves, kinky, mhmm, opinions, truely, nick, jealousy, prank,
# Nearest to no: solver, e-books, taught, matter, guarantee, lies, drastic, willrries,