import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords') # stopwords 
from nltk.corpus import stopwords 
import tensorflow as tf
from gensim.models import Doc2Vec

class doc_MLP():

    def __init__(self, 
                addr_data='../data/recall_labeled.csv'):

        # load in the labeled samples
        self.data_labeled = pd.read_csv(addr_data)

        

    # This function isolates the labeled samples from the 
    # full dataset of labeled and unlabeled samples
    def isolate_labeled(self):
        df = pd.read_csv('../data/recall.csv')
        data_labeled = df.ix[~np.isnan(df['SC'])]
        data_labeled.to_csv('../data/recall_labeled.csv', index=None)

dmlp = doc_MLP()
dmlp.isolate_labeled()