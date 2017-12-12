from helpers import *
import numpy as np

# path_positive = "twitter-datasets/train_pos_full.txt"
# path_negative = "twitter-datasets/train_neg_full.txt"
#
# numWords = []
# positive_files_total = []
# negative_files_total = []
#
# with open(path_positive, "r") as f:
#     for line in f:
#         positive_files_total.append(line)
#         counter = len(line.split())
#         numWords.append(counter)
#
#
# with open(path_negative, "r", encoding='utf-8') as f:
#     for line in f:
#         negative_files_total.append(line)
#         counter = len(line.split())
#         numWords.append(counter)
#
# words_list = create_word_list2(positive_files_total + negative_files_total)
# wordsList = np.save('words_list_15.npy', words_list)
wordsList = np.load('words_list_15.npy')

print(wordsList[:100])
print('\n')
print(len(wordsList))
