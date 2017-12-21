# EPFL Machine Learning Course CS-433, Fall 2017, first project

## Description

These scripts concern themselves with natural language processing, within the field of machine learning. The task at hand is to analyse short sentences, so called tweets and predict whether the sentence has a positive or negative sentiment. This task is solved through implementing some simple preprocessing, creating word embeddings through the skip-gram architecture and finally training three neural networks in order to evaluate which one gives the best results with this specific task. The implemented neural networks are Long Short-Term Memory on its own, and in a modified version where it is preceded by convolutional neural network. In the latter case, two editions are presented, one with a fixed kernel size, and another with a double kernel size.

## Getting Started

First of all, make sure you have installed:

1. python 3.5 or higher

2. tensorFlow. Here you have the link to the official [installation guide](https://www.tensorflow.org/install/)

3. Keras. Link with all [installation steps](https://keras.io/#installation)

4. Gensim. (Gensim is needed only if you want to reproduce our code for the preprocessing in the script "skipgram_gensim.py"), here you have a link for [installation](https://radimrehurek.com/gensim/install.html)

After this, be sure to download the directory "data" from this [link on switch drive](https://drive.switch.ch/index.php/s/FvBb3jbLUNBN1Bl). Copy the downloaded directory at the same level of this readMe file!

### Description of "data" Directory

In the 'data' directory we have:

1. 'combined_tweets' folder that contains a txt file with all the tweets in one file. We used this file in the preprocessing script "skipgram_gensim".

2. 'downloaded_word_vectors' folder that contains all the necessary files to use the word vectors that we downloaded from glove. Specifically it contains the wordVectors list, the dictionary list, the ids matrix of the train data and the ids matrix of the test data.

3. 'our_trained_word_vectors' directory, in which we have all the files to use our word vectors created during the preprocessing. We have the word vector list, the words list (set of all words used), the ids matrix of train data and the ids matrix of the test data.

4. 'twitter-datasets' directory, in which we have all the tweets data downloaded from Kaggle.

5. 'models' directory, where we have the models computed with the "LSTM_kaggle_score_0.85620.py" script

6. 'kaggle_score_0.8594' directory, in which we have the model and wheights of the model that obtained our best score

7. 'kaggle_score_0.8548' directory, where you can find the model and the model wheights of the second best score with the two kernels CNN LSTM architecture.
 

## description of the scripts

In the 'scripts' directory we have three folders, one for each of the architectures that we built for our project. In each of these architecture we have the script where we build and train our architecture (we created three different folder for the architectures because the scripts will create files about metrics and models and we want to have them separated). Here you have a description of the three folders:

1. 'best_score_two_kernels_CNN_LSTM': here you can find the script that obtained our best Kaggle score (85.94%), the 'run.py' script. In this script we build our CNN_LSTM with two kernel sizes for the convolutional layer using Keras; we train the model using our word vectors obtained in our pre-processing (see the "skipgram_gensim.py" script) and finally we create a .csv file for kaggle submission. If you want to obtain the .csv file, just run this script, a file will be crated at the same level of the script

2. 'LSTM_best' folder: Here you can find the script where we build the LSTM architecture using TensorFLow.

3. 'one_kernel_CNN_LSTM_best_score' folder: In this folder we have the script where we create our architecture CNN_LSTM with one kernel size, we train the model and then we create a .csv file for kaggle submission.

At the same level of the 'scripts' directory we have the following files:

1. 'helpers.py' file: in this file we have all the utility methods that we created for our project. We have methods to load the data, methods useful for the pre-processing part, for building layers of our architectures and for the submission.

2. 'skipgram_gensim.py' file: in this script we create the word vectors using the skip-gram process, more info in the description inside the file.

3. 'make_submission_tf.py' file: this script is needed in order to create a valid submission using the LSTM architecture built with TensorFlow (not Keras, more info below)


## Running the scripts

1. In order to obtain the .csv file for the best score that we had on Kaggle, you have to run the script "run.py" in the folder with relative path "scripts/best_score_two_kernels_CNN_LSTM". When this script was run, a windows machine was used (tensorFlow for windows comes compiled in a non-optimized version for the latest CPU) so the running time of all process was nearly 20 hours. If you have GPU version of tensorFlow you can reduce running time drastically. However, if you do not want to spent so much time, you can modify the value of the variable "trainable" to False (by default is True) at the line 41. With this change you will obtain our second highest score (85.48%) with a running time of 2 hours and a half on a windows machine and CPU version of tensorFlow. The .csv file will be at the same level of the 'run.py' script.

2. If you want to try the architecture with one kernel size CNN_LSTM, run the 'kaggle_score_0.8526_one_kernel_CNN_LSTM.py' in the folder 'scripts/one_kernel_CNN_LSTM_best_score'. At the end you will find the .csv file at the same level of the script.


3. In order to try the LSTM architecture, run the 'LSTM_kaggle_score_0.85620.py' script in the folder 'scripts/LSTM_best'. After you run it, you have to run the 'make_submission_tf.script' (you can find it at the same level of the 'scritps' folder), this script will create a .csv file for the submission.

4. If you want to test the the pre-processing part where we create the word vectors and the ids matrix for train and test data, run the 'skipgram_gensim.py' script(you need to have gensim installed for this task).

5. if you do not want to rebuild all the models for the different architectures, in the 'data' folder you can find the json files of the models and the weights of the model for the different scores. In order to reproduce the .csv file, just create a short python script where you call the 'make_submission' method from our 'helpers' file. Be sure to pass the right argument to the method (the only thing that you have to do is just call the 'make_submission' method). Here it is a code example for the prediction script:

```python
from helpers import *

keras_prediction(model_path="data/kaggle_score_0.8594/kaggle_score_0.8594_model.json", weights_path="data/kaggle_score_0.8594/kaggle_score_0.8594__weights.h5", ids_test_path="../../data/our_trained_wordvectors/ids_test_sg_6.npy", csv_file_name="run_prediction.csv")
```

## Authors

* **Eigil Lippert** [eigil-lippert](https://github.com/eigil-lippert)
* **Christian Sciuto** [christian-5-28](https://github.com/christian-5-28)
* **Lorenzo Tarantino** [lorenzotara](https://github.com/lorenzotara)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


