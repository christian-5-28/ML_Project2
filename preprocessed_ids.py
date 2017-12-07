import numpy as np
import re
from random import randint
from nltk.tokenize import word_tokenize
import tensorflow as tf
import matplotlib.pyplot as plt

#Preprocessing all tweets and creating ids, took 25h....
def tokenize(s):
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
    string = tokenize(string)

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

num_files_total = len(numWords)
print('The total number of files is', num_files_total)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

max_seq_length = 20

'''loading and creating word vectors and dictionary'''
glove_model_path = '/Users/eyu/Google Drev/DTU/5_semester/ML/ML_Project2/glove/glove.twitter.27B.25d.txt'

f = open(glove_model_path, 'r')

counter = 0

word_list = []
word_vectors = []

for line in f:
    counter += 1

    splitted_line = line.split()

    if len(splitted_line[1:]) != 25:
        continue

    word_list.append(splitted_line[0])
    word_vectors.append(splitted_line[1:])

f.close()

positive_files = positive_files_total
negative_files = negative_files_total
num_files_mini = len(positive_files) + len(negative_files)

'''
Convert to an ids matrix
'''
ids = np.zeros((num_files_mini, max_seq_length), dtype='int32')
file_counter = 0
for line in positive_files:
    index_counter = 0
    split = clean_sentences(line)  # Cleaning the sentence

    for word in split:
        try:
            ids[file_counter][index_counter] = word_list.index(word)
        except ValueError:
            ids[file_counter][index_counter] = 18  # Vector for unknown positive vectors, not used otherwise
        index_counter = index_counter + 1

        # If we have already seen maxSeqLength words, we break the loop of the words of a tweet
        if index_counter >= max_seq_length:
            break
    file_counter = file_counter + 1

    print("Steps to end: " + str(len(positive_files) + len(negative_files) - file_counter))


for line in negative_files:
    index_counter = 0
    split = clean_sentences(line)

    for word in split:
        try:
            ids[file_counter][index_counter] = word_list.index(word)
        except ValueError:
            ids[file_counter][index_counter] = 19  # Vector for unknown negative vectors, not used otherwise
        index_counter = index_counter + 1

        if index_counter >= max_seq_length:
            break
    file_counter = file_counter + 1

    print("Steps to end: " + str(len(positive_files) + len(negative_files) - file_counter))


np.save('pp_ids_train_tweet_matrix.npy', ids)

ids = np.load('pp_ids_train_tweet_matrix.npy')
print(ids.shape)