# ------------------------------------------------------------
# S e a B a s s 
# 2 0 1 9
# 
# Util.py
# 
# Description - Used to support the Medical Device Recalls 
# Data Classification project.
# ------------------------------------------------------------

import nltk
nltk.download('stopwords')  # stopwords 
nltk.download('wordnet')    # lemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 

import pandas as pd
import numpy as np

# used to convert sentance to list of words
# this removes stop words, punctuation, and numbers
def tokenize(sentance):

    # my stopwords that i find
    # TODO find scientific method of detecting these with 
    # cross class frequency analysis
    my_stopwords = ['may', 'result', 'potential', 'patient']

    # Remove punctuation and digits
    c_rm = '!"#$%&\'(),-.:;?@[]^`{|}~0123456789'   # Replace these with ''
    c_rp = '*+/<=>\\_'                              # Replace these with ' '
    sentance_clean = ''
    for c in sentance:
        if not c in c_rm:
            if c in c_rp:
                sentance_clean += ' '
            else:
                sentance_clean += c

    # convert the setance to a list of words
    words_unfiltered = word_tokenize(sentance_clean, language='english')

    # load the stop words from nltk (a, the, and...)
    stop_words = set(stopwords.words('english'))

    # load the lammatizer (Rocks -> rock)
    lemmatizer = WordNetLemmatizer()

    # remove stop words and miss-spelled words
    words = []
    for word in words_unfiltered:
        if (word.lower() not in stop_words) and (word.lower() not in my_stopwords):
            words.append(lemmatizer.lemmatize(word.lower()))

    return words

# Utility function to seperate the labeled vs unlabeled data
def isolate_labeled():
    
    # read in the labeled and unlabeled dataset
    df = pd.read_csv('../data/recall.csv')

    # isolate the two
    df_labeled = df.ix[~np.isnan(df['SC'])]
    df_unlabeled = df.ix[np.isnan(df['SC'])]

    # write them back to respective files
    df_labeled.to_csv('../data/recall_labeled.csv', index=None)
    df_unlabeled.to_csv('../data/recall_unlabeled.csv', index=None)
