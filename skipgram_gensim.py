from gensim import corpora, models, similarities

import os
import logging
from keras import layers
from keras import models as mods
from helpers import *
import io


def get_sim(valid_word_idx, vocab_size):
    """
    finds similarity score for the current word, compared all other words.
    outputs similarity score for every word in descending order
    """

    sim = np.zeros((vocab_size,))
    in_arr1 = np.zeros((1,))
    in_arr2 = np.zeros((1,))
    for i in range(vocab_size):
        in_arr1[0,] = valid_word_idx
        in_arr2[0,] = i
        out = validation_model.predict_on_batch([in_arr1, in_arr2])
        sim[i] = out
    return sim


class MySentences(object):
    """
    Iterator object that iterates through files in directory, picking every sentence from the file.
    This is why "combined_full.txt" should be placed in its own directory.
    """
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in io.open(os.path.join(self.dirname, fname), 'r', encoding="utf-8", errors="replace"):
                yield clean_sentences(line)


'''Loading senctences in a memory-friendly way, needs full path'''
sentences = MySentences("data/combined_tweets")  # a memory-friendly iterator

'''For logging the process'''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''Gensim model computation, either load existing or compute from scratch'''
vector_dim = 300  # dimensions of word vectors = 300
model = models.word2vec.Word2Vec(sentences, sg=1, iter=15, min_count=6, size=vector_dim, workers=4, negative=5)

# uncomment to load instead of computing:
# model = models.Word2Vec.load("/Users/eyu/Google Drev/DTU/5_semester/ML/ML_Project2/gensim models/model_TEST")

'''Generate word list from gensim model'''
# Iterates through every word vector from the model, and extracts the corresponding word.
word_vecs = np.zeros((len(model.wv.vocab), vector_dim))
dictionary = []
indices = []
for i in range(len(model.wv.vocab)):
    vector = model.wv[model.wv.index2word[i]]
    if vector is None:
        print('none: ', model.wv.index2word[i])
    if vector is not None:
        word_vecs[i] = vector
        dictionary.append(model.wv.index2word[i])

# Unknown word vector for unknown words, with the token UNK added as last token in array:
word_vecs = np.vstack((word_vecs, [np.random.uniform(-1, 1) for elem in np.zeros(vector_dim)]))
print(word_vecs.shape)
dictionary.append('UNK')


'''Saving'''
# Saves the model, word vectors and word list
model.save("model_sg_6")
np.save('wordvecs_sg_6.npy', word_vecs)
np.save('word_list_sg_6.npy', dictionary)


'''Validation of the similiar words with Keras (for qualitative analysis)'''
# Creates the embedding matrix of of the word list from the skip-gram model
embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

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


del model, embedding_matrix

'''CREATE IDS'''
path_positive = "data/twitter-datasets/train_pos_full.txt"
path_negative = "data/twitter-datasets/train_neg_full.txt"
path_test = "data/twitter-datasets/test_data.txt"


numWords = []
positive_files_total = []
negative_files_total = []
test_files_total = []

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

with open(path_test, "r") as f:
    for line in f:
        test_files_total.append(line)
        counter = len(line.split())
        numWords.append(counter)
print('Test files finished')

del path_positive, path_negative

num_files_total = len(numWords)
print('The total number of files is', num_files_total)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))


max_seq_length = 20
positive_files = positive_files_total
negative_files = negative_files_total
test_files = test_files_total

total_files_length = len(positive_files) + len(negative_files)

# creating the ids matrix for our train data
create_ids_matrix(positive_files, negative_files, max_seq_length, dictionary)

# finally from here, we create the ids matrix on our test data file
path_test = "data/twitter-datasets/test_data.txt"
test_files = []

with open(path_test, "r") as f:
    for line in f:
        test_files.append(line)

max_seq_length = 20

indices = []

num_files_mini = len(test_files)
total_files_length = len(test_files)

'''
Convert to an ids matrix
'''

dictionary = np.load('data/our_trained_wordvectors/word_list_sg_6.npy')
dictionary = dictionary.tolist()

ids = np.zeros((num_files_mini, max_seq_length), dtype='int32')
file_counter = 0
start_time = datetime.datetime.now()

for line in test_files:
    index_counter = 0
    comma_index = line.index(',')
    line = line[comma_index+1:]
    split = clean_sentences(line)  # Cleaning the sentence

    for word in split:
        try:
            ids[file_counter][index_counter] = dictionary.index(word)
        except ValueError:
            ids[file_counter][index_counter] = len(dictionary) - 1  # Vector for unknown positive vectors, not used otherwise
        index_counter = index_counter + 1

        # If we have already seen maxSeqLength words, we break the loop of the words of a tweet
        if index_counter >= max_seq_length:
            break

    if file_counter % 1000 == 0:
        print("Steps to end: " + str(total_files_length - file_counter))
        print('Time of execution: ', datetime.datetime.now() - start_time)

    file_counter = file_counter + 1

np.save('data/our_trained_wordvectors/ids_test_sg_6.npy', ids)
