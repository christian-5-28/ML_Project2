import numpy as np
word_vectors = np.loadtxt('data/wordvecs.txt', usecols=range(300), dtype=float)
print(len(word_vectors[0]))
print(type(word_vectors[0][0]))