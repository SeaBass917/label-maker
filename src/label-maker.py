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
    
# For converting strings to indecies
# TODO try to remove this, its only being 
# the dictionary indexing is much cleaner
class Classes(Enum):
    Security = 0
    Hardware = 1
    Software = 2

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

        # load the local data files
        self.load_data()
        self.load_weights()

        # if any of the metadata is missing recalculate it
        if( not self.keywords.get('N_SC') or
            not self.keywords.get('N_HW') or
            not self.keywords.get('N_SW') or
            not self.keywords.get('N_SCHW') or
            not self.keywords.get('N_SCSW') or
            not self.keywords.get('N_HWSW') or
            not self.keywords.get('N_SCHWSW') or
            not self.keywords.get('N_OTHER') or
            not self.keywords.get('N_TOT') 
        ):
            print("\tWarning: Could not find metadata in dictionary. Recalculating...")

            # initialize the metadata
            self.keywords['N_SC'] = 0
            self.keywords['N_HW'] = 0
            self.keywords['N_SW'] = 0
            self.keywords['N_SCHW'] = 0
            self.keywords['N_SCSW'] = 0
            self.keywords['N_HWSW'] = 0
            self.keywords['N_SCHWSW'] = 0
            self.keywords['N_OTHER'] = 0
            self.keywords['N_TOT'] = 0

            # loop through the labeled data and recalc the metainfo
            for _, data in self.data_labeled.iterrows():
                self.keywords['N_TOT'] += 1
                self.keywords['N_SC'] += data.loc['SC']
                self.keywords['N_HW'] += data.loc['HW']
                self.keywords['N_SW'] += data.loc['SW']
                if(data.loc['SC'] == 1 and data.loc['HW'] == 1):
                    self.keywords['N_SCHW'] += 1
                if(data.loc['SC'] == 1 and data.loc['SW'] == 1):
                    self.keywords['N_SCSW'] += 1
                if(data.loc['HW'] == 1 and data.loc['SW'] == 1):
                    self.keywords['N_HWSW'] += 1
                if(data.loc['SC'] == 1 and data.loc['HW'] == 1 and data.loc['SW'] == 1):
                    self.keywords['N_SCHWSW'] += 1
                if(data.loc['SC'] == 0 and data.loc['HW'] == 0 and data.loc['SW'] == 0):
                    self.keywords['N_OTHER'] += 1

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
        
        # new keywords dict
        self.keywords = {}

        # loop through the labeled dataset
        # recount the frequencies for the keywords
        for _, data in self.data_labeled.iterrows():

            sentance = self.tokenize(data.loc['MANUFACTURER_RECALL_REASON'])
            for word in sentance:
                if not word in self.keywords:
                    self.keywords[word] = {'SC':0, 'HW':0, 'SW':0, 'TOT':0}

                self.keywords[word]['SC'] += data.loc['SC']
                self.keywords[word]['HW'] += data.loc['HW']
                self.keywords[word]['SW'] += data.loc['SW']
                self.keywords[word]['TOT'] += 1
        
        self.save_weights()

    # update the keywords and their weights 
    # with the new list of words with their label
    def weight_update(self, words):

        # loop through the new words adjusting 
        # their weights using the label
        for word in words:

            # extract the weight tuple
            weight = self.keywords.get(word)

            # if new word we need non null kw
            if weight is None:
                weight = {'SC':0, 'HW':0, 'SW':0, 'TOT':0}

            # add the new labels to the prev total
            # note the labels are taken from the gui checkbox values which are 0 or 1
            weight['SC'] += self.var_sc.get()
            weight['HW'] += self.var_hw.get()
            weight['SW'] += self.var_sw.get()
            weight['TOT'] += 1

            # store updated weight back in the keyword table
            self.keywords[word] = weight

        # save the weights
        self.save_weights()

    # calculate the full weight for a list of words
    # this weight is almost a prediction of sorts
    # bound between [0,1]
    # SUM(weight(word))/n_words
    def sample_weight(self, words):

        # sum the weights
        sample_label = np.array([0,0,0], dtype='float32')

        # use these to normalize the weights
        # theyre isolated so we can ignore indecisive keywords
        # (weight ~= 0.5)
        norm_SC = 0.0
        norm_HW = 0.0
        norm_SW = 0.0

        # loop through the words in the recall
        for word in words:

            # look up the freuencies in the table
            freqs = self.keywords.get(word)

            # if the word isn't in the table, stay all zeros
            if freqs is not None:
                
                # start a weight array with the keyword class frequencies
                weights = np.array([
                                    freqs['SC'],
                                    freqs['HW'],
                                    freqs['SW'],
                                    freqs['TOT']
                                    ], dtype='float32')

                # calculate the ratio of words in each class to total word frequency
                if(weights[2]):
                    weights[0] /= weights[2]
                else:
                    weights[0] /= weights[3]
                weights[1] /= weights[3]
                weights[2] /= weights[3]

                # TODO Explore more complex weight functions
                # Currently just summing and normalizing the ratios
                # - try doubling weights that are near 0 or 1
                # - try ignoring weights (0.4, 0.6) as these words are ambiguous
                
                #if(weights[0] < 0.4 or 0.6 < weights[0]):
                sample_label[0] += weights[0]
                norm_SC += 1
                #if(weights[1] < 0.4 or 0.6 < weights[1]):
                sample_label[1] += weights[1]
                norm_HW += 1
                #if(weights[2] < 0.4 or 0.6 < weights[2]):
                sample_label[2] += weights[2]
                norm_SW += 1
        
        # normalize
        if(norm_SC):
            sample_label[0] /= norm_SC
        if(norm_HW):
            sample_label[1] /= norm_HW
        if(norm_SW):
            sample_label[2] /= norm_SW
        
        return sample_label

    # technically its a method
    def dont_call_this_function(self):
        missing_the_magic_word = True
        while(missing_the_magic_word):
            print('ah ah ah, you didn\'t say the magic word')

    # save the dictionary to a local file
    def save_weights(self):
        with open(self.addr_weights, 'wb') as f:
            pk.dump(self.keywords, f, pk.HIGHEST_PROTOCOL)

    # read the dictionary from a local file
    def load_weights(self):
        try:
            with open(self.addr_weights, 'rb') as f:
                self.keywords = pk.load(f)
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
            if word.lower() not in stop_words:
                words.append(lemmatizer.lemmatize(word.lower()))

        return words
        words

    # find the index of the sample whose weight is closest to the weight specified
    def search_weight(self, weight=1.0):

        # initialize min weight difference and idx
        weight_diff_min = 1
        idx_max = 0

        # check the dropdown box to see what class were looking for
        trait = Classes[self.search_label.get()].value
        
        # loop through a random subset of the dataset
        for _ in range(min(512, self.data_unlabeled.shape[0])):
            
            # grab sample at random
            i = int(np.random.uniform(0, self.data_unlabeled.shape[0]))
                
            # get sample -> tokenize -> calculate weight
            sample_label = self.sample_weight(self.tokenize(self.data_unlabeled.loc[i,'MANUFACTURER_RECALL_REASON']))

            # update weight max
            weight_diff = abs(weight - sample_label[trait])
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
        if(abs(self.var_sc.get() - self.weight_cur[0]) < 0.5):
            self.SC_correct += 1
        if(abs(self.var_hw.get() - self.weight_cur[1]) < 0.5):
            self.HW_correct += 1
        if(abs(self.var_sw.get() - self.weight_cur[2]) < 0.5):
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
        self.keywords['N_SC'] += self.var_sc.get()
        self.keywords['N_HW'] += self.var_hw.get()
        self.keywords['N_SW'] += self.var_sw.get()
        if(self.var_sc.get() and self.var_hw.get()):
            self.keywords['N_SCHW'] += 1
        if(self.var_sc.get() and self.var_sw.get()):
            self.keywords['N_SCSW'] += 1
        if(self.var_hw.get() and self.var_sw.get()):
            self.keywords['N_HWSW'] += 1
        if(self.var_sc.get() and self.var_hw.get() and self.var_sw.get()):
            self.keywords['N_SCHWSW'] += 1
        self.keywords['N_TOT'] += 1

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
        self.sample_count_label['text'] = 'Samples labeled: ' + str(self.keywords['N_TOT']) + "(" + str(self.samples_labeled_this_run) + ")"
        self.SC_count_label['text'] = "Security: " + str(self.keywords['N_SC']) 
        self.HW_count_label['text'] = "Hardware: " + str(self.keywords['N_HW'])
        self.SW_count_label['text'] = "Software: " + str(self.keywords['N_SW'])
        self.SC_label['text'] = ': ' + str(self.weight_cur[0])
        self.HW_label['text'] = ': ' + str(self.weight_cur[1])
        self.SW_label['text'] = ': ' + str(self.weight_cur[2])

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
        
        self.weight_cur = self.sample_weight(self.tokenize(self.sentance))

        # update the model state for the user
        self.sample_label['text'] = 'Sample: ' + str(self.idx_cur)[:7]
        self.sample_count_label['text'] = 'Samples labeled: ' + str(self.keywords['N_TOT']) + "(" + str(self.samples_labeled_this_run) + ")"
        self.SC_count_label['text'] = "Security: " + str(self.keywords['N_SC']) 
        self.HW_count_label['text'] = "Hardware: " + str(self.keywords['N_HW'])
        self.SW_count_label['text'] = "Software: " + str(self.keywords['N_SW'])
        self.SC_label['text'] = ': ' + str(self.weight_cur[0])
        self.HW_label['text'] = ': ' + str(self.weight_cur[1])
        self.SW_label['text'] = ': ' + str(self.weight_cur[2])

        # post it in the box
        self.text_box.delete('1.0', tk.END)
        self.text_box.insert(tk.END, self.sentance)

        # uncheck the checkboxes
        self.var_sc.set(0)
        self.var_hw.set(0)
        self.var_sw.set(0)

    # Find the top 100 keywords for each class
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

        for kw in self.keywords:

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
            if(self.keywords[kw]['SC'] > lowest_freqs['SC'][1]):
                SC_top[lowest_freqs['SC'][0]][0] = kw
                SC_top[lowest_freqs['SC'][0]][1] = self.keywords[kw]['SC']
            if(self.keywords[kw]['HW'] > lowest_freqs['HW'][1]):
                HW_top[lowest_freqs['HW'][0]][0] = kw
                HW_top[lowest_freqs['HW'][0]][1] = self.keywords[kw]['HW']
            if(self.keywords[kw]['SW'] > lowest_freqs['SW'][1]):
                SW_top[lowest_freqs['SW'][0]][0] = kw
                SW_top[lowest_freqs['SW'][0]][1] = self.keywords[kw]['SW']
        
        #print('Security')
        #print(SC_top)
        #print('Hardware')
        #print(HW_top)
        #print('Software')
        #print(SW_top)

        # sort them before you return them
        return (SC_top.sort(key = lambda x: x[1]), HW_top.sort(key = lambda x: x[1]), SW_top.sort(key = lambda x: x[1]))

    # for all top 15-50 keywords evaluate the grep approach
    def sweep_grep(self):

        # generate top 100 keywords for each class
        # sorted from highest to lowest frequency
        SC_keywords, HW_keywords, SW_keywords = self.top_keyword_get()

        # loop through the various 
        keyword_count = 15
        while (keyword_count <= 50):

            # limit the keywords to the top 'keyword_count' keywords
            SC_keywords_ltd = SC_keywords[0:keyword_count-1]
            HW_keywords_ltd = HW_keywords[0:keyword_count-1]
            SW_keywords_ltd = SW_keywords[0:keyword_count-1]



            keyword_count+=1

    # test the performance of the model based on the labeled samples
    def test(self, keywords_HW, keywords_SW, keywords_SC):

        print("Evaluating performance...")

        # accuracies for the weighted dictionary and grep approaches
        SC_acc_WD = 0.0
        SW_acc_WD = 0.0
        HW_acc_WD = 0.0
        SC_acc_grep = 0.0
        SW_acc_grep = 0.0
        HW_acc_grep = 0.0
        labeled_count = 0

        # for each labeled sample
        for _, data in self.data_labeled.iterrows():

            # increment number of total labeled samples
            labeled_count += 1

            # tokenize the sentance
            sentence = self.tokenize(data.loc['MANUFACTURER_RECALL_REASON'])

            # classify the sample using the grep approach
            SC = 0.0
            for word in sentence:
                if(word in keywords_SC):
                    SC = 1.0
            if(abs(data.loc['SC'] - SC) < 0.5):
                SC_acc_grep += 1

            HW = 0.0
            for word in sentence:
                if(word in keywords_HW):
                    HW = 1.0
            if(abs(data.loc['HW'] - HW) < 0.5):
                HW_acc_grep += 1

            SW = 0.0
            for word in sentence:
                if(word in keywords_SW):
                    SW = 1.0
            if(abs(data.loc['SW'] - SW) < 0.5):
                SW_acc_grep += 1

            # classify using the weighted dictionary
            sample_label = self.sample_weight(sentence)

            if(abs(data.loc['SC'] - sample_label[0]) < 0.5):
                SC_acc_WD += 1
            if(abs(data.loc['HW'] - sample_label[1]) < 0.5):
                HW_acc_WD += 1
            if(abs(data.loc['SW'] - sample_label[2]) < 0.5):
                SW_acc_WD += 1
        
        SC_acc_WD /= labeled_count
        HW_acc_WD /= labeled_count
        SW_acc_WD /= labeled_count
        SC_acc_grep /= labeled_count
        HW_acc_grep /= labeled_count
        SW_acc_grep /= labeled_count

        print("Accuracy for Security Threats: WD [", SC_acc_WD, "] vs grep [", SC_acc_grep, "]")
        print("Accuracy for Hardware Issues : WD [", HW_acc_WD, "] vs grep [", HW_acc_grep, "]")
        print("Accuracy for Software Issues : WD [", SW_acc_WD, "] vs grep [", SW_acc_grep, "]")

    # print out a summary of the model
    def summary(self):
        print("Class Frequencies\n",
        "Security: ", self.keywords['N_SC'],    "\n",
        "Hardware: ", self.keywords['N_HW'],    "\n",
        "Software: ", self.keywords['N_SW'],    "\n",
        "SC&HW:    ", self.keywords['N_SCHW'],  "\n",
        "SC&SW:    ", self.keywords['N_SCSW'],  "\n",
        "HW&SW:    ", self.keywords['N_HWSW'],  "\n",
        "SC&HW&SW: ", self.keywords['N_SCHWSW'],"\n",
        "Other:    ", self.keywords['N_OTHER'], "\n",
        "Total:    ", self.keywords['N_TOT'],   "\n")

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

        print(float(len(SC_SW_overlap)) / float(len(SC_top)) * 100, '%% overlap between SC and SW keywords')

        print('The following words are unique to Security threats')
        for SC_word_freq in SC_top:
            if(SC_word_freq[0] not in SC_SW_overlap):
                print(SC_word_freq[0], SC_word_freq[1], '/', self.keywords[SC_word_freq[0]]['TOT'])


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
        self.search_label.set('Security')
        drop_down = tk.OptionMenu(main_window, self.search_label, *{'Security', 'Hardware', 'Software'})
        drop_down.configure(bg='#404040', fg='#ffffff')
        drop_down['menu'].configure(bg='#404040', fg='#ffffff')
        drop_down.grid(row=9, column=0)
        
        # submit the vote/label and get the next sample
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='Submit\nGet Confident Sample', width=25, command=self.submit_confident).grid(row=9, column=1)
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='Submit\nGet Unconfident Sample', width=25, command=self.submit_unconfident).grid(row=9, column=2)

        # Allow user to search for keywords in unlabeled samples
        self.search_key = tk.StringVar()
        self.search_key.set("Battery...Software...Problem...")
        tk.Label(main_window, text="Keyword Search", bg='#404040', fg='#ffffff').grid(row=10, column=0)
        tk.Entry(main_window, textvariable=self.search_key, bg='#202020', fg='#ffffff').grid(row=10, column=1)
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='Search', width=25, command=self.search_keyword).grid(row=10, column=2)

        # call now to load the next(first) sample
        self.next()

        # put it up
        main_window.mainloop()

isolate_labeled()
l = Labeler()
l.run()