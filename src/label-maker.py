import tkinter as tk
import pandas as pd
import numpy as np
import pickle as pk
from enum import Enum
import string
import nltk
nltk.download('stopwords')  # stopwords 
nltk.download('wordnet')    # lemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 

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

class Labeler():

    # initialize the vocabulary with the address of the corpus we will be labeling
    def __init__(self, 
                addr_labeled_data='../data/recall_labeled.csv', 
                addr_unlabeled_data='../data/recall_unlabeled.csv', 
                addr_weights='../data/weighted-dictionary.pk1'):

        # location in filesystem for the datasets
        self.addr_labeled_data = '../data/recall_labeled.csv'
        self.addr_unlabeled_data = '../data/recall_unlabeled.csv'
        self.addr_weights = addr_weights
        
        # my stopwords that i find
        # TODO find scientific method of detecting these with 
        # cross class frequency analysis
        self.my_stopwords = ['may', 'result', 'potential', 'patient']

        # load the local data files
        self.load_data()
        self.load_weights()

        # set the metadata label
        # so we know to ignore it
        self.metadata_labels = [
            'N_SC', 'N_HW', 'N_SW',
            'N_SCHW', 'N_SCSW', 'N_HWSW',
            'N_SCHWSW', 'N_OTHER', 'N_TOT'
        ]

        # if any of the metadata is missing recalculate it
        if( not self.weighted_dict.get('N_SC') or not self.weighted_dict.get('N_HW') or
            not self.weighted_dict.get('N_SW') or not self.weighted_dict.get('N_SCHW') or
            not self.weighted_dict.get('N_SCSW') or not self.weighted_dict.get('N_HWSW') or
            not self.weighted_dict.get('N_SCHWSW') or not self.weighted_dict.get('N_OTHER') or
            not self.weighted_dict.get('N_TOT') 
        ):
            print("\tWarning: Could not find metadata in dictionary. Recalculating...")

            # initialize the metadata
            self.weighted_dict['N_SC'] = 0
            self.weighted_dict['N_HW'] = 0
            self.weighted_dict['N_SW'] = 0
            self.weighted_dict['N_SCHW'] = 0
            self.weighted_dict['N_SCSW'] = 0
            self.weighted_dict['N_HWSW'] = 0
            self.weighted_dict['N_SCHWSW'] = 0
            self.weighted_dict['N_OTHER'] = 0
            self.weighted_dict['N_TOT'] = 0

            # loop through the labeled data and recalc the metainfo
            for _, data in self.data_labeled.iterrows():
                self.weighted_dict['N_TOT'] += 1
                self.weighted_dict['N_SC'] += data.loc['SC']
                self.weighted_dict['N_HW'] += data.loc['HW']
                self.weighted_dict['N_SW'] += data.loc['SW']
                if(data.loc['SC'] == 1 and data.loc['HW'] == 1):
                    self.weighted_dict['N_SCHW'] += 1
                if(data.loc['SC'] == 1 and data.loc['SW'] == 1):
                    self.weighted_dict['N_SCSW'] += 1
                if(data.loc['HW'] == 1 and data.loc['SW'] == 1):
                    self.weighted_dict['N_HWSW'] += 1
                if(data.loc['SC'] == 1 and data.loc['HW'] == 1 and data.loc['SW'] == 1):
                    self.weighted_dict['N_SCHWSW'] += 1
                if(data.loc['SC'] == 0 and data.loc['HW'] == 0 and data.loc['SW'] == 0):
                    self.weighted_dict['N_OTHER'] += 1
            
            # update with the new metadata
            self.save_weights()

        # init vars used for real time perfomance analysis
        self.samples_labeled_this_run = 0
        self.SC_correct = 0
        self.HW_correct = 0
        self.SW_correct = 0

        # alemzadeh's keywords
        self.alem_keywords_HW = [   'board', 'chip', 'hardware', 'processor', 
                                    'memory', 'disk', 'PCB', 'electronic', 
                                    'electrical', 'circuit', 'leak', 'short-circuit', 
                                    'capacitor', 'transistor', 'resistor', 'battery', 
                                    'power', 'supply', 'outlet', 'plug', 'power-up', 
                                    'discharge', 'charger']
        self.alem_keywords_SW = [   'software', 'application', 'function', 'code', 
                                    'version', 'backup', 'database', 'program', 
                                    'bug', 'java', 'run', 'upgrade']

    # for when I edit the dataset manually
    def refresh_weights(self):
        
        # new weighted_dict dict fom the
        self.weighted_dict = self.get_weighted_dict(self.data_labeled)
        
        # save the weights locally
        self.save_weights()
    
    # Generate a weighted dictionary from a labeled dataset 
    def get_weighted_dict(self, X):
        
        # new weighted_dict dict
        weighted_dict = {}

        # loop through the labeled dataset
        # recount the frequencies for the weighted_dict
        for _, data in X.iterrows():

            sentance = self.tokenize(data.loc['MANUFACTURER_RECALL_REASON'])
            for word in sentance:
                if not word in weighted_dict:
                    weighted_dict[word] = {'SC':0, 'HW':0, 'SW':0, 'TOT':0}

                weighted_dict[word]['SC'] += data.loc['SC']
                weighted_dict[word]['HW'] += data.loc['HW']
                weighted_dict[word]['SW'] += data.loc['SW']
                weighted_dict[word]['TOT'] += 1
        
        return weighted_dict

    # update the weighted_dict and their weights 
    # with the new list of words with their label
    def weight_update(self, words):

        # loop through the new words adjusting 
        # their weights using the label
        for word in words:

            # extract the weight tuple
            weight = self.weighted_dict.get(word)

            # if new word we need non null kw
            if weight is None:
                weight = {'SC':0, 'HW':0, 'SW':0, 'TOT':0}

            # add the new labels to the prev total
            # note the labels are taken from the gui checkbox values which are 0 or 1
            weight['SC'] += self.var_sc.get()
            weight['HW'] += self.var_hw.get()
            weight['SW'] += self.var_sw.get()
            weight['TOT'] += 1

            # store updated weight back in the weighted dictionary
            self.weighted_dict[word] = weight

        # save the weights
        self.save_weights()

    # calculate the full weight for a list of words
    # this weight is almost a prediction of sorts
    # bound between [0,1]
    # SUM(weight(word))/n_words
    def weight_sentance(self, sentance, weighted_dict):
        
        words = self.tokenize(sentance)

        # sum the weights
        sample_weight = {'SC': 0.0, 'HW': 0.0, 'SW': 0.0}

        # use these to normalize the weights
        # theyre isolated so we can ignore indecisive weighted_dict
        # (weight ~= 0.5)
        norm_SC = 0.0
        norm_HW = 0.0
        norm_SW = 0.0

        # loop through the words in the recall
        for word in words:

            # look up the frequencies in the table
            freqs = weighted_dict.get(word)

            # if the word isn't in the table, stay all zeros
            if freqs is not None:

                # TODO Explore more complex weight functions
                # Currently just summing and normalizing the ratios
                # - try doubling weights that are near 0 or 1
                # - try ignoring weights (0.4, 0.6) as these words are ambiguous

                # calculate the ratio of words in each class to total word frequency
                if(freqs['SC'] < 0.5 or 0.5 < freqs['SC']):
                    if(freqs['SW']):
                        sample_weight['SC'] += float(freqs['SC']) / freqs['SW']
                    else:
                        sample_weight['SC'] += float(freqs['SC']) / freqs['TOT']
                    norm_SC += 1
                if(freqs['HW'] < 0.5 or 0.5 < freqs['HW']):
                    sample_weight['HW'] += float(freqs['HW']) / freqs['TOT']
                    norm_HW += 1
                if(freqs['HW'] < 0.5 or 0.5 < freqs['HW']):
                    sample_weight['SW'] += float(freqs['SW']) / freqs['TOT']
                    norm_SW += 1
                
        
        # normalize
        if(norm_SC):
            sample_weight['SC'] /= norm_SC
        if(norm_HW):
            sample_weight['HW'] /= norm_HW
        if(norm_SW):
            sample_weight['SW'] /= norm_SW
        
        return sample_weight

    # technically its a method
    def dont_call_this_function(self):
        missing_the_magic_word = True
        while(missing_the_magic_word):
            print('ah ah ah, you didn\'t say the magic word')

    # save the dictionary to a local file
    def save_weights(self):
        with open(self.addr_weights, 'wb') as f:
            pk.dump(self.weighted_dict, f, pk.HIGHEST_PROTOCOL)

    # read the dictionary from a local file
    def load_weights(self):
        try:
            with open(self.addr_weights, 'rb') as f:
                self.weighted_dict = pk.load(f)
        except:
            print('\tWarning: Dictionary not found, recalculating weights...')
            self.refresh_weights()

    # save the labeled and unlabeled data to local storage
    def save_data(self):
        self.data_labeled.to_csv(self.addr_labeled_data, index=None)
        self.data_unlabeled.to_csv(self.addr_unlabeled_data, index=None)

    # read the labeled and unlabeled data from local storage
    def load_data(self):
        self.data_labeled = pd.read_csv(self.addr_labeled_data)
        self.data_unlabeled = pd.read_csv(self.addr_unlabeled_data)

    # used to convert sentance to list of words
    # this removes stop words, punctuation, and numbers
    def tokenize(self, sentance):

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
            if (word.lower() not in stop_words) and (word.lower() not in self.my_stopwords):
                words.append(lemmatizer.lemmatize(word.lower()))

        return words
        words

    # find the index of the sample whose weight is closest to the weight specified
    def search_weight(self, weight=1.0):

        # initialize min weight difference and idx
        weight_diff_min = 1
        idx_max = 0
        
        # loop through a random subset of the dataset
        for _ in range(min(512, self.data_unlabeled.shape[0])):
            
            # grab sample at random
            i = int(np.random.uniform(0, self.data_unlabeled.shape[0]))
                
            # get sample -> tokenize -> calculate weight
            sample_label = self.weight_sentance(self.data_unlabeled.loc[i,'MANUFACTURER_RECALL_REASON'], self.weighted_dict)

            # update weight max
            weight_diff = abs(weight - sample_label[self.search_label.get()])
            if(weight_diff < weight_diff_min):
                weight_cur = sample_label
                weight_diff_min = weight_diff
                idx_max = i

        return idx_max, weight_cur
    
    # commit a user specified label to the dictionary
    def submit(self):

        # calculate real time performance based on the submitted label
        # and the models prediction
        self.samples_labeled_this_run += 1
        if(abs(self.var_sc.get() - self.weight_cur['SC']) < 0.5):
            self.SC_correct += 1
        if(abs(self.var_hw.get() - self.weight_cur['HW']) < 0.5):
            self.HW_correct += 1
        if(abs(self.var_sw.get() - self.weight_cur['SW']) < 0.5):
            self.SW_correct += 1
        
        self.SC_perf['text'] = str(float(self.SC_correct) / float(self.samples_labeled_this_run))
        self.HW_perf['text'] = str(float(self.HW_correct) / float(self.samples_labeled_this_run))
        self.SW_perf['text'] = str(float(self.SW_correct) / float(self.samples_labeled_this_run))

        # label this sample -> add it to the labeled data -> drop it from the unlabeled data
        self.data_unlabeled.iloc[self.idx_cur,-3] = self.var_sc.get()
        self.data_unlabeled.iloc[self.idx_cur,-2] = self.var_hw.get()
        self.data_unlabeled.iloc[self.idx_cur,-1] = self.var_sw.get()
        self.data_labeled = self.data_labeled.append(self.data_unlabeled.iloc[self.idx_cur], ignore_index=True)
        self.data_unlabeled = self.data_unlabeled.drop(index=self.idx_cur)

        
        # update how many labeled sampled there are
        self.weighted_dict['N_SC'] += self.var_sc.get()
        self.weighted_dict['N_HW'] += self.var_hw.get()
        self.weighted_dict['N_SW'] += self.var_sw.get()
        if(self.var_sc.get() and self.var_hw.get()):
            self.weighted_dict['N_SCHW'] += 1
        if(self.var_sc.get() and self.var_sw.get()):
            self.weighted_dict['N_SCSW'] += 1
        if(self.var_hw.get() and self.var_sw.get()):
            self.weighted_dict['N_HWSW'] += 1
        if(self.var_sc.get() and self.var_hw.get() and self.var_sw.get()):
            self.weighted_dict['N_SCHWSW'] += 1
        self.weighted_dict['N_TOT'] += 1

        # save the data updates to local storage
        self.save_data()

        # tokenize the current sentance were looking at
        words = self.tokenize(self.sentance)

        # update the weights based on the new label
        self.weight_update(words)

    # submit + get next sample with weight closest to 1.0
    def submit_confident(self):
        self.submit()
        self.next(weight=1.0)
    
    # submit + get next sample with weight closest to 0.5
    def submit_unconfident(self):
        self.submit()
        self.next(weight=0.5)

    # search for a sample with the classification closest to 'weight'
    def next(self, weight=1.0):

        # find next sample
        self.idx_cur, self.weight_cur = self.search_weight(weight=weight)

        # update the model state for the user
        self.sample_label['text'] = 'Sample: ' + str(self.idx_cur)[:7]
        self.sample_count_label['text'] = 'Samples labeled: ' + str(self.weighted_dict['N_TOT']) + "(" + str(self.samples_labeled_this_run) + ")"
        self.SC_count_label['text'] = "Security: " + str(self.weighted_dict['N_SC']) 
        self.HW_count_label['text'] = "Hardware: " + str(self.weighted_dict['N_HW'])
        self.SW_count_label['text'] = "Software: " + str(self.weighted_dict['N_SW'])
        self.SC_label['text'] = ': ' + str(self.weight_cur['SC'])
        self.HW_label['text'] = ': ' + str(self.weight_cur['HW'])
        self.SW_label['text'] = ': ' + str(self.weight_cur['SW'])

        # paste current recall into the textbox
        self.sentance = self.data_unlabeled.loc[self.idx_cur,'MANUFACTURER_RECALL_REASON']
        self.text_box.delete('1.0', tk.END)
        self.text_box.insert(tk.END, self.sentance)

        # uncheck the checkboxes
        self.var_sc.set(0)
        self.var_hw.set(0)
        self.var_sw.set(0)

    # search for the next sample that contains the user specified keyword
    def search_keyword(self):

        # get search parameter from text box
        keyword = self.search_key.get()

        # only searches one word at a time
        # just cut off other stuff
        keyword = keyword.split()[0] 

        # get first unlabeled sample with a keyword match
        match = False
        for i, data in self.data_unlabeled.iterrows():
            self.sentance = data.loc['MANUFACTURER_RECALL_REASON']
            if keyword in self.sentance:
                match = True
                self.idx_cur = i
                break

        # if there is no match then set these
        if not match:
            i = -1
            self.sentance = "<No Keyword Match>"
        
        self.weight_cur = self.weight_sentance(self.sentance, self.weighted_dict)

        # update the model state for the user
        self.sample_label['text'] = 'Sample: ' + str(self.idx_cur)[:7]
        self.sample_count_label['text'] = 'Samples labeled: ' + str(self.weighted_dict['N_TOT']) + "(" + str(self.samples_labeled_this_run) + ")"
        self.SC_count_label['text'] = "Security: " + str(self.weighted_dict['N_SC']) 
        self.HW_count_label['text'] = "Hardware: " + str(self.weighted_dict['N_HW'])
        self.SW_count_label['text'] = "Software: " + str(self.weighted_dict['N_SW'])
        self.SC_label['text'] = ': ' + str(self.weight_cur['SC'])
        self.HW_label['text'] = ': ' + str(self.weight_cur['HW'])
        self.SW_label['text'] = ': ' + str(self.weight_cur['SW'])

        # post it in the box
        self.text_box.delete('1.0', tk.END)
        self.text_box.insert(tk.END, self.sentance)

        # uncheck the checkboxes
        self.var_sc.set(0)
        self.var_hw.set(0)
        self.var_sw.set(0)

    # Find the top 100 weighted_dict for each class
    def top_keyword_get(self):
        
        # use tuples to represent each class:
        # (SC, SW, HW)
        SC_top = [  ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0],
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0],
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0],
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0]]
        HW_top = [  ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0],
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0],
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0],
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0]]
        SW_top = [  ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0],
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0],
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0],
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0], 
                    ["", 0], ["", 0], ["", 0], ["", 0], ["", 0]]

        lowest_freqs = {'SC' : (0, 8675309), 'HW' : (0, 8675309), 'SW' : (0, 8675309)}

        # loop through kewords ignoring metadata
        for kw in self.weighted_dict:
            if(kw not in self.metadata_labels):
                # refresh current min for each class
                # initialize to monitor the element with the lowest frequency and what that frequency is; in the top 15's
                # rather than re-sorting the list everytime im just gunna keep track of where the min is
                # python is weird and too flexable, just init to a big number
                lowest_freqs = {'SC' : (0, 8675309), 'HW' : (0, 8675309), 'SW' : (0, 8675309)}
                for i, (SC, HW, SW) in enumerate(zip(SC_top, HW_top, SW_top)):
                    if(SC[1] <= lowest_freqs['SC'][1]):
                        lowest_freqs['SC'] = (i, SC[1])
                    if(HW[1] <= lowest_freqs['HW'][1]):
                        lowest_freqs['HW'] = (i, HW[1])
                    if(SW[1] <= lowest_freqs['SW'][1]):
                        lowest_freqs['SW'] = (i, SW[1])
                
                # if the current keyword frequency is greater than 
                # the lowest frequency keyword for that class then
                # replace the lowest with the current
                if(self.weighted_dict[kw]['SC'] > lowest_freqs['SC'][1]):
                    SC_top[lowest_freqs['SC'][0]][0] = kw
                    SC_top[lowest_freqs['SC'][0]][1] = self.weighted_dict[kw]['SC']
                if(self.weighted_dict[kw]['HW'] > lowest_freqs['HW'][1]):
                    HW_top[lowest_freqs['HW'][0]][0] = kw
                    HW_top[lowest_freqs['HW'][0]][1] = self.weighted_dict[kw]['HW']
                if(self.weighted_dict[kw]['SW'] > lowest_freqs['SW'][1]):
                    SW_top[lowest_freqs['SW'][0]][0] = kw
                    SW_top[lowest_freqs['SW'][0]][1] = self.weighted_dict[kw]['SW']
        
        #print('Security')
        #print(SC_top)
        #print('Hardware')
        #print(HW_top)
        #print('Software')
        #print(SW_top)

        # sort them before you return them
        SC_top.sort(key = lambda x: x[1], reverse=True)
        HW_top.sort(key = lambda x: x[1], reverse=True)
        SW_top.sort(key = lambda x: x[1], reverse=True)
        return (SC_top, HW_top, SW_top)
    
    # classify using the grep approach
    def clf_grep(self, sentance, weighted_dict_SC, weighted_dict_HW, weighted_dict_SW):

        # tokenize the sentance
        words = self.tokenize(sentance)

        # initialize the classifications
        classifications = {'SC': 0, 'HW': 0, 'SW': 0}

        # for each class look for weighted_dict, 
        # flag a classification on a keyword match
        # NOTE: weighted_dict is a list of tuples
        for word in words:
            for kw in weighted_dict_SC:
                if(word == kw[0]):
                    classifications['SC'] = 1
            for kw in weighted_dict_HW:
                if(word == kw[0]):
                    classifications['HW'] = 1
            for kw in weighted_dict_SW:
                if(word == kw[0]):
                    classifications['SW'] = 1

        return classifications

    # classify using the weighted dictionary approach
    def clf_WD(self, sentance):

        # initialize the classifications
        classifications = {'SC': 0, 'HW': 0, 'SW': 0}

        # get the weights for each class
        weights = self.weight_sentance(sentance, self.weighted_dict)

        # Boundary on 0.5
        if(weights['SC'] < 0.5):
            classifications['SC'] = 0
        else:
            classifications['SC'] = 1
        if(weights['HW'] < 0.5):
            classifications['HW'] = 0
        else:
            classifications['HW'] = 1
        if(weights['SW'] < 0.5):
            classifications['SW'] = 0
        else:
            classifications['SW'] = 1

        return classifications

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

    # test the performance of the model based on the labeled samples
    def test(self, fold_count=10):

        sample_per_fold = int(self.data_labeled.shape[0] / fold_count)

        print("Evaluating performance...")
        print(fold_count, "fold cross validation;", sample_per_fold, "samples per fold.")

        # sample all the rows at random (shuffling the dataset)
        X = self.data_labeled.sample(frac=1.0) 

        accuracies = pd.DataFrame.from_dict({
                'SC_test' : np.zeros(fold_count),
                'SC_train' : np.zeros(fold_count),
                'HW_test' : np.zeros(fold_count),
                'HW_train' : np.zeros(fold_count),
                'SW_test' : np.zeros(fold_count),
                'SW_train' : np.zeros(fold_count)
            },
        orient='index')

        for fold in range(fold_count):

            print("Fold:", fold+1, "-----")
            min_idx = fold*sample_per_fold
            max_idx = (fold + 1) * sample_per_fold

            # grab a fold of the data
            X_test = X.iloc[min_idx : max_idx, : ]

            # the rest is training data
            X_train = X.iloc[0 : min_idx, : ].append(X.iloc[max_idx : -1, : ], ignore_index=True)

            # get a new dictionary
            print("\t Calculating weights...", end='')
            weighted_dict = self.get_weighted_dict(X_train)
            print("Done.")

            # keep track of how many predictions were correct
            correct = {
                'SC': 0.0,
                'HW': 0.0,
                'SW': 0.0
            }

            # for each labeled test sample
            print("\t Testing performance on the testing set...", end='')
            for _, data in X_test.iterrows():

                # get the sentance
                sentence = data.loc['MANUFACTURER_RECALL_REASON']

                # calculate the weight for this sample
                sample_label = self.weight_sentance(sentence, weighted_dict)

                # Compare the weight to the labele to determine correctness
                if(abs(data.loc['SC'] - sample_label['SC']) < 0.5):
                    correct['SC'] += 1.0
                if(abs(data.loc['HW'] - sample_label['HW']) < 0.5):
                    correct['HW'] += 1.0
                if(abs(data.loc['SW'] - sample_label['SW']) < 0.5):
                    correct['SW'] += 1.0
            
            # normalize and store in the dataframe
            accuracies.loc['SC_test'][fold] = correct['SC'] / X_test.shape[0]
            accuracies.loc['HW_test'][fold] = correct['HW'] / X_test.shape[0]
            accuracies.loc['SW_test'][fold] = correct['SW'] / X_test.shape[0]
            print("Done.")

            # keep track of how many predictions were correct
            correct = {
                'SC': 0.0,
                'HW': 0.0,
                'SW': 0.0
            }

            # for each labeled training sample
            print("\t Testing performance on the training set...", end='')
            for _, data in X_train.iterrows():

                # get the sentance
                sentence = data.loc['MANUFACTURER_RECALL_REASON']

                # calculate the weight for this sample
                sample_label = self.weight_sentance(sentence, weighted_dict)

                # Compare the weight to the labele to determine correctness
                if(abs(data.loc['SC'] - sample_label['SC']) < 0.5):
                    correct['SC'] += 1.0
                if(abs(data.loc['HW'] - sample_label['HW']) < 0.5):
                    correct['HW'] += 1.0
                if(abs(data.loc['SW'] - sample_label['SW']) < 0.5):
                    correct['SW'] += 1.0
            
            # normalize and store in the dataframe
            accuracies.loc['SC_train'][fold] = correct['SC'] / X_train.shape[0]
            accuracies.loc['HW_train'][fold] = correct['HW'] / X_train.shape[0]
            accuracies.loc['SW_train'][fold] = correct['SW'] / X_train.shape[0]
            print("Done.")
        
        # done, return the dataframe
        return accuracies

    # print out a summary of the model
    def summary(self):
        print("Class Frequencies\n",
        "Security: ", self.weighted_dict['N_SC'],    "\n",
        "Hardware: ", self.weighted_dict['N_HW'],    "\n",
        "Software: ", self.weighted_dict['N_SW'],    "\n",
        "SC&HW:    ", self.weighted_dict['N_SCHW'],  "\n",
        "SC&SW:    ", self.weighted_dict['N_SCSW'],  "\n",
        "HW&SW:    ", self.weighted_dict['N_HWSW'],  "\n",
        "SC&HW&SW: ", self.weighted_dict['N_SCHWSW'],"\n",
        "Other:    ", self.weighted_dict['N_OTHER'], "\n",
        "Total:    ", self.weighted_dict['N_TOT'],   "\n")

    # Test to determine the keyword overlap between software and security
    def determine_SC_SW_overlap(self):
        (SC_top, _, SW_top) = self.top_keyword_get()

        SC_SW_overlap = []

        print('These words are the overlap between SC and SW')
        for SC_word_freq in SC_top:
            for SW_word_freq in SW_top:
                if(SC_word_freq[0] == SW_word_freq[0]):
                    SC_SW_overlap.append(SC_word_freq[0])
                    print('SC', SC_word_freq, '\tSW', SW_word_freq)

        print(float(len(SC_SW_overlap)) / float(len(SC_top)) * 100, '%% overlap between SC and SW weighted_dict')

        print('The following words are unique to Security threats')
        for SC_word_freq in SC_top:
            if(SC_word_freq[0] not in SC_SW_overlap):
                print(SC_word_freq[0], SC_word_freq[1], '/', self.weighted_dict[SC_word_freq[0]]['TOT'])


        return SC_SW_overlap

    # start running the program
    def run(self):
        
        ## initialize a gui for this

        # main window
        main_window = tk.Tk()
        main_window.title('Label Maker v3.0')
        main_window.configure(bg='#404040')
 
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='>', height=25, command=self.next).grid(rowspan=10, column=4)
        
        # label to say the current sample number (like an id)
        self.sample_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.sample_label.grid(row=0, column=0, sticky=tk.W)

        # Idsplay number of abeled samples, and specify each classes frequency
        self.sample_count_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.SC_count_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.HW_count_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.SW_count_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.sample_count_label.grid(row=1, column=0, sticky=tk.W)
        self.SC_count_label.grid(row=2, column=0, sticky=tk.W)
        self.HW_count_label.grid(row=3, column=0, sticky=tk.W)
        self.SW_count_label.grid(row=4, column=0, sticky=tk.W)
        
        # display a real time performance calc
        self.SC_perf = tk.Label(main_window, text="<>", bg='#404040', fg='#ffffff')
        self.HW_perf = tk.Label(main_window, text="<>", bg='#404040', fg='#ffffff')
        self.SW_perf = tk.Label(main_window, text="<>", bg='#404040', fg='#ffffff')
        self.SC_perf.grid(row=2, column=1, sticky=tk.W)
        self.HW_perf.grid(row=3, column=1, sticky=tk.W)
        self.SW_perf.grid(row=4, column=1, sticky=tk.W)

        # check boxes for voting
        self.var_sc = tk.IntVar()
        self.var_hw = tk.IntVar()
        self.var_sw = tk.IntVar()
        tk.Checkbutton(main_window, bg='#404040', fg='#ffffff', selectcolor="#202020", text='Security', variable=self.var_sc).grid(row=5, column=0, sticky=tk.W)
        tk.Checkbutton(main_window, bg='#404040', fg='#ffffff', selectcolor="#202020", text='Hardware', variable=self.var_hw).grid(row=6, column=0, sticky=tk.W)
        tk.Checkbutton(main_window, bg='#404040', fg='#ffffff', selectcolor="#202020", text='Software', variable=self.var_sw).grid(row=7, column=0, sticky=tk.W)

        # labels for the checkboxes
        self.SC_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.HW_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.SW_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.SC_label.grid(row=5, column=1, sticky=tk.W)
        self.HW_label.grid(row=6, column=1, sticky=tk.W)
        self.SW_label.grid(row=7, column=1, sticky=tk.W)

        # this is where the text will be pasted
        self.text_box = tk.Text(main_window, bg='#202020', fg='#ffffff', width=50, height=10)
        self.text_box.grid(row=8, columnspan=3)

        # let the user decide what class we want to focus on
        self.search_label = tk.StringVar()
        self.search_label.set('SC')
        drop_down = tk.OptionMenu(main_window, self.search_label, *{'SC', 'HW', 'SW'})
        drop_down.configure(bg='#404040', fg='#ffffff')
        drop_down['menu'].configure(bg='#404040', fg='#ffffff')
        drop_down.grid(row=9, column=0)
        
        # submit the vote/label and get the next sample
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='Submit\nGet Confident Sample', width=25, command=self.submit_confident).grid(row=9, column=1)
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='Submit\nGet Unconfident Sample', width=25, command=self.submit_unconfident).grid(row=9, column=2)

        # Allow user to search for weighted_dict in unlabeled samples
        self.search_key = tk.StringVar()
        self.search_key.set("Battery...Software...Problem...")
        tk.Label(main_window, text="Keyword Search", bg='#404040', fg='#ffffff').grid(row=10, column=0)
        tk.Entry(main_window, textvariable=self.search_key, bg='#202020', fg='#ffffff').grid(row=10, column=1)
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='Search', width=25, command=self.search_keyword).grid(row=10, column=2)

        # call now to load the next(first) sample
        self.next()

        # put it up
        main_window.mainloop()

l = Labeler()
#accs = l.test(fold_count=8)
#accs.to_csv('../data/8-fold_X-val.csv')
l.run()