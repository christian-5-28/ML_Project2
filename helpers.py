import pandas as pd
import numpy as np
import re
import datetime


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
        cleaned_line = clean_sentences(line)  # Cleaning the sentence
        split = cleaned_line.split()

        for word in split:
            try:
                ids[file_counter][index_counter] = wordsList.index(word)
            except ValueError:
                ids[file_counter][index_counter] = len(wordsList)  # Vector for unkown words
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

