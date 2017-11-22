import pandas as pd


def load_data():
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

    lexicons = pd.read_table('lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', sep='\t',
                             names=('term', 'emotion', 'value'))

    terms = lexicons.term

    emotions = lexicons.drop('term', axis=1).pivot(columns='emotion', values='value')
    emotions['term'] = terms
    emotions.fillna(0, inplace=True)
    emotions = emotions.groupby('term').sum()
    emotions.reset_index(inplace=True)

    return emotions

