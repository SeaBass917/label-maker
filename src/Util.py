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

# TODO: Update this, it's currently using old datastructures
"""
# for all top 1-100 weighted_dict evaluate the grep approach
def sweep_grep(self):

    print(" --- Sweeping the grep approach from 1 - 100 top weighted_dict. --- ")

    # generate top 100 weighted_dict for each class
    # sorted from highest to lowest frequency
    SC_weighted_dict, HW_weighted_dict, SW_weighted_dict = self.top_keyword_get()

    # array of accuracies for each level we're sweeping
    # for each class of interest
    accuracies = pd.DataFrame.from_dict({  
        'SC': np.zeros(100), 
        'HW': np.zeros(100),
        'SW': np.zeros(100)
    }, orient='index')

    # loop through each keyword limit
    for keyword_count in range(100):

        print(keyword_count, "weighted_dict...")

        # limit the weighted_dict to the top 'keyword_count' weighted_dict
        SC_weighted_dict_ltd = SC_weighted_dict[0:keyword_count+1]
        HW_weighted_dict_ltd = HW_weighted_dict[0:keyword_count+1]
        SW_weighted_dict_ltd = SW_weighted_dict[0:keyword_count+1]

        correct = {
            'SC': 0.0,
            'HW': 0.0,
            'SW': 0.0
        }

        # for each labeled sample
        for _, data in self.data_labeled.iterrows():

            # classify by grep approach
            classifications = self.clf_grep(data.loc['MANUFACTURER_RECALL_REASON'], SC_weighted_dict_ltd, HW_weighted_dict_ltd, SW_weighted_dict_ltd)

            # update the accuracy counter
            if(abs(data.loc['SC'] - classifications['SC']) < 0.5):
                correct['SC'] += 1.0
            if(abs(data.loc['HW'] - classifications['HW']) < 0.5):
                correct['HW'] += 1.0
            if(abs(data.loc['SW'] - classifications['SW']) < 0.5):
                correct['SW'] += 1.0

        # divide by total samples to get accuracy
        # and store that into the dataframe
        accuracies.loc['SC', keyword_count] = correct['SC'] / self.data_labeled.shape[0]
        accuracies.loc['HW', keyword_count] = correct['HW'] / self.data_labeled.shape[0]
        accuracies.loc['SW', keyword_count] = correct['SW'] / self.data_labeled.shape[0]
    
    # one last round with just the almezedah weighted_dict
    correct = {
            'HW': 0.0,
            'SW': 0.0
    }

    # for each labeled sample
    for _, data in self.data_labeled.iterrows():

        # classify by grep approach
        classifications = self.clf_grep(data.loc['MANUFACTURER_RECALL_REASON'], [], self.alem_keywords_HW, self.alem_keywords_SW)

        # update the accuracy counter
        if(abs(data.loc['HW'] - classifications['HW']) < 0.5):
            correct['HW'] += 1.0
        if(abs(data.loc['SW'] - classifications['SW']) < 0.5):
            correct['SW'] += 1.0

    accuracies.loc['Alem_HW', 0] = correct['HW'] / self.data_labeled.shape[0]
    accuracies.loc['Alem_SW', 0] = correct['SW'] / self.data_labeled.shape[0]

    # return the accuracies calculated
    return accuracies
"""