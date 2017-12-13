import numpy as np
from helpers import clean_sentences_eigil

#Preprocessing all tweets and creating ids, took 25h....

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

del path_positive, path_negative

num_files_total = len(numWords)
print('The total number of files is', num_files_total)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

max_seq_length = 20


positive_files = positive_files_total
negative_files = negative_files_total
num_files_mini = len(positive_files) + len(negative_files)
total_files_length = len(positive_files) + len(negative_files)
'''
Convert to an ids matrix
'''
ids = np.zeros((num_files_mini, max_seq_length), dtype='int32')
file_counter = 0
for line in positive_files:
    index_counter = 0
    split = clean_sentences_eigil(line)  # Cleaning the sentence

    for word in split:
        try:
            ids[file_counter][index_counter] = word_list.index(word) + 1
        except ValueError:
            ids[file_counter][index_counter] = 18  # Vector for unknown positive vectors, not used otherwise
        index_counter = index_counter + 1

        # If we have already seen maxSeqLength words, we break the loop of the words of a tweet
        if index_counter >= max_seq_length:
            break


del positive_files

for line in negative_files:
    index_counter = 0
    split = clean_sentences_eigil(line)

    for word in split:
        try:
            ids[file_counter][index_counter] = word_list.index(word) + 1
        except ValueError:
            ids[file_counter][index_counter] = 19  # Vector for unknown negative vectors, not used otherwise
        index_counter = index_counter + 1

        if index_counter >= max_seq_length:
            break



np.save('pp_sg_ids_matrix.npy', ids)

ids = np.load('pp_sg_ids_train_matrix.npy')
print(ids.shape)


# # TODO: TRASH:
# '''loading and creating word vectors and dictionary'''
# glove_model_path = 'for_eigil/glove/glove.twitter.27B.25d.txt'
#
# f = open(glove_model_path, 'r')
#
# counter = 0
#
# word_list = []
# word_vectors = []
#
# for line in f:
#     counter += 1
#
#     splitted_line = line.split()
#
#     if len(splitted_line[1:]) != 25:
#         continue
#
#     word_list.append(splitted_line[0])
#     word_vectors.append(splitted_line[1:])
#
# f.close()