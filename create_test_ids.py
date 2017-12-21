"""
Create ids matrix for test_set
"""

from helpers import *
import datetime

'''CREATE IDS'''
path_test = "twitter-datasets/test_data.txt"
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

dictionary = np.load('skipgrams/word_list_sg_2.npy')
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


np.save('skipgrams/ids_test_sg_2.npy', ids)

ids = np.load('skipgrams/ids_test_sg_2.npy')
print(ids[:4])
print(ids.shape)
