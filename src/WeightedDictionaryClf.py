# ------------------------------------------------------------
# S e a B a s s 
# 2 0 1 9
# 
# WeightedDictionaryClf.py
# 
# Description - This classifier is a weighted dictionary. The 
# dictionary stores the frequencies of each word used in 
# classifying FDA recall data. Those frequencies are then used 
# to determine the overall weight of a sentance amoung the 
# three classes: Security Threats, Hardware Faults, and 
# Software Faults.
# ------------------------------------------------------------
from Util import tokenize
from Util import sigmoid
import pickle as pk
import numpy as np
import pandas as pd

class WD():

    # 
    def __init__(self, 
                addr_weights='../data/weighted-dictionary.pk1',
                alpha=0):
        
        self.addr_weights = addr_weights

        # alpha in the sigmoid
        # 0 for no sigmoid
        self.alpha = alpha

        # set the metadata label
        # so we know to ignore it
        self.metadata_labels = [
            'N_SC', 'N_HW', 'N_SW',
            'N_SCHW', 'N_SCSW', 'N_HWSW',
            'N_SCHWSW', 'N_OTHER', 'N_TOT'
        ]

        self.weighted_dict = {}
        self.my_stopwords_SC = []
        self.my_stopwords_HW = []
        self.my_stopwords_SW = []

    # overload the brackets to just index the internal dictionary
    def __getitem__(self, key):
        return self.weighted_dict[key]
    def __setitem__(self, key, item):
        self.weighted_dict[key] = item

    # pass through the get method
    def get(self, key):
        return self.weighted_dict.get(key)

    # save the dictionary to a local file
    def save_weights(self, addr_weights='../data/weighted-dictionary.pk1'):

        # Use the class addr if it is non null
        save_addr = ''
        if(self.addr_weights):
            save_addr = self.addr_weights
        else:
            save_addr = addr_weights

        with open(save_addr, 'wb') as f:
            pk.dump(self.weighted_dict, f, pk.HIGHEST_PROTOCOL)

    # read the dictionary from a local file
    def load_weights(self, addr_weights='../data/weighted-dictionary.pk1'):

        # Use the class addr if it is non null
        load_addr = ''
        if(self.addr_weights):
            load_addr = self.addr_weights
        else:
            load_addr = addr_weights

        success = False
        try:
            with open(load_addr, 'rb') as f:
                self.weighted_dict = pk.load(f)
                self.my_stopwords_SC, self.my_stopwords_HW, self.my_stopwords_SW = self.get_stop_words()   # get stopwords at this point
                success = True
        except:
            print('\tWarning!: No dictionary found at: \"', load_addr, '\".')
            self.weighted_dict = {}

        return success

    # return the metadata
    def get_metadata(self):

        # Verify that the dictionary was already loaded
        if len(self.weighted_dict):
            
            return {
                'N_SC': self.weighted_dict['N_SC'],
                'N_HW': self.weighted_dict['N_HW'],
                'N_SW': self.weighted_dict['N_SW'],
                'N_SCHW': self.weighted_dict['N_SCHW'],
                'N_SCSW': self.weighted_dict['N_SCSW'],
                'N_HWSW': self.weighted_dict['N_HWSW'],
                'N_SCHWSW': self.weighted_dict['N_SCHWSW'],
                'N_OTHER': self.weighted_dict['N_OTHER'],
                'N_TOT': self.weighted_dict['N_TOT']
            }

        else:
            print('\tError! -- dictionary is empty. Try: load_weights() or fit()?')
            return None

    # for when I edit the dataset manually
    def refresh_weights(self, data_labeled):
        
        # new weighted_dict dict fom the
        self.weighted_dict = self.get_weighted_dict(data_labeled)

        # refresh stopwords while im here
        self.my_stopwords_SC, self.my_stopwords_HW, self.my_stopwords_SW = self.get_stop_words()   # get stopwords at this point
                
    # Generate a weighted dictionary from a labeled dataset 
    def get_weighted_dict(self, data_labeled):
        
        # new weighted_dict
        weighted_dict = {}

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

        # loop through the labeled dataset
        # recount the frequencies for the weighted_dict
        for _, data in data_labeled.iterrows():

            # metadata calculations
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

            # weight calculations
            sentance = tokenize(data.loc['MANUFACTURER_RECALL_REASON'])
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
    def update_dict(self, weighted_dict, sentance, y_SC, y_HW, y_SW):

        words = tokenize(sentance)

        # loop through the new words adjusting 
        # their weights using the label
        for word in words:

            # extract the weight tuple
            weight = weighted_dict.get(word)

            # if new word we need non null kw
            if weight is None:
                weight = {'SC':0, 'HW':0, 'SW':0, 'TOT':0}

            # add the new labels to the prev total
            # note the labels are taken from the gui checkbox values which are 0 or 1
            weight['SC'] += y_SC
            weight['HW'] += y_HW
            weight['SW'] += y_SW
            weight['TOT'] += 1

            # store updated weight back in the weighted dictionary
            weighted_dict[word] = weight

        return weighted_dict

    def update_model(self, sentance, y_SC, y_HW, y_SW):

        # update dictionary
        self.weighted_dict = self.update_dict(self.weighted_dict, sentance, y_SC, y_HW, y_SW)

        # save the weights
        self.weighted_dict.save_weights()

    # calculate the full weight for a list of words
    # this weight is almost a prediction of sorts
    # bound between [0,1]
    # sig(SUM(weight(word))/n_words)
    def weight_sentance(self, sentance, weighted_dict):
        
        words_unclean = tokenize(sentance)

        words_SC = [word for word in words_unclean if word not in self.my_stopwords_SC]
        words_HW = [word for word in words_unclean if word not in self.my_stopwords_HW]
        words_SW = [word for word in words_unclean if word not in self.my_stopwords_SW]

        # sum the weights
        sample_weight = {'SC': 0.0, 'HW': 0.0, 'SW': 0.0}

        # use these to normalize the weights
        norm_SC = 0.0
        norm_HW = 0.0
        norm_SW = 0.0

        # Security weightings
        for word in words_SC:

            # look up the frequencies in the table
            freqs = weighted_dict.get(word)

            # if the word is in the table
            if freqs is not None:

                # TODO: Not sure if referencing SW is helping
                if(freqs['SW']):
                    sample_weight['SC'] += float(freqs['SC']) / freqs['SW']
                else:
                    sample_weight['SC'] += float(freqs['SC']) / freqs['TOT']
                norm_SC += 1

        # Hardware weightings
        for word in words_HW:

            # look up the frequencies in the table
            freqs = weighted_dict.get(word)

            # if the word is in the table
            if freqs is not None:
                sample_weight['HW'] += float(freqs['HW']) / freqs['TOT']
                norm_HW += 1

        # Software weightings
        for word in words_SW:

            # look up the frequencies in the table
            freqs = weighted_dict.get(word)

            # if the word is in the table
            if freqs is not None:

                sample_weight['SW'] += float(freqs['SW']) / freqs['TOT']
                norm_SW += 1
                
        # normalize
        if(norm_SC):
            sample_weight['SC'] /= norm_SC
        if(norm_HW):
            sample_weight['HW'] /= norm_HW
        if(norm_SW):
            sample_weight['SW'] /= norm_SW

        # Map to sigmoid
        if(self.alpha != 0):
            sample_weight['SC'] = sigmoid(sample_weight['SC'], alpha=self.alpha)
            sample_weight['HW'] = sigmoid(sample_weight['HW'], alpha=self.alpha)
            sample_weight['SW'] = sigmoid(sample_weight['SW'], alpha=self.alpha)
        
        return sample_weight

    # Find the top 100 weighted_dict for each class
    def top_keyword_get(self, verbose=False):
        
        # 100 blank tuples for the top
        # ("word", freq)
        SC_top = [ ("", 0) for _ in range(100)]
        HW_top = [ ("", 0) for _ in range(100)]
        SW_top = [ ("", 0) for _ in range(100)]

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
                    SC_top[lowest_freqs['SC'][0]] = (kw, self.weighted_dict[kw]['SC'])
                if(self.weighted_dict[kw]['HW'] > lowest_freqs['HW'][1]):
                    HW_top[lowest_freqs['HW'][0]] = (kw, self.weighted_dict[kw]['HW'])
                if(self.weighted_dict[kw]['SW'] > lowest_freqs['SW'][1]):
                    SW_top[lowest_freqs['SW'][0]] = (kw, self.weighted_dict[kw]['SW'])
        
        # sort them
        SC_top.sort(key = lambda x: x[1], reverse=True)
        HW_top.sort(key = lambda x: x[1], reverse=True)
        SW_top.sort(key = lambda x: x[1], reverse=True)

        # print them to the console
        if(verbose):
            print('Security')
            print(SC_top)
            print('Hardware')
            print(HW_top)
            print('Software')
            print(SW_top)

        # return them
        return (SC_top, HW_top, SW_top)
    
    # determine a list of all words 
    # for each class that have between a 0.4 and 0.6 weight
    def get_stop_words(self, min_weight=0.45, max_weight=0.55, verbose=False):

        # Empty list for each class
        SC_stopwords = []
        HW_stopwords = []
        SW_stopwords = []

        # loop through kewords ignoring metadata
        for kw in self.weighted_dict:
            if(kw not in self.metadata_labels):

                SC_weight = 0
                HW_weight = 0
                SW_weight = 0
                
                # Determine the weights
                SC_weight = float(self.weighted_dict[kw]['SC']) / float(self.weighted_dict[kw]['TOT'])
                HW_weight = float(self.weighted_dict[kw]['HW']) / float(self.weighted_dict[kw]['TOT'])
                SW_weight = float(self.weighted_dict[kw]['SW']) / float(self.weighted_dict[kw]['TOT'])

                # If they are between min and max add them to the stop words
                if(min_weight < SC_weight and SC_weight < max_weight):
                    SC_stopwords.append(kw)
                if(min_weight < HW_weight and HW_weight < max_weight):
                    HW_stopwords.append(kw)
                if(min_weight < SW_weight and SW_weight < max_weight):
                    SW_stopwords.append(kw)

        # print them to the console
        if(verbose):
            print('Security(', len(SC_stopwords), ')')
            print(SC_stopwords)
            print('Hardware(', len(HW_stopwords), ')')
            print(HW_stopwords)
            print('Software(', len(SW_stopwords), ')')
            print(SW_stopwords)

        # return them
        return (SC_stopwords, HW_stopwords, SW_stopwords)

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

    # test the performance of the model based on the labeled samples
    def test(self, data_labeled, fold_count=8):

        sample_per_fold = int(data_labeled.shape[0] / fold_count)

        print("Evaluating performance...")
        print(fold_count, "fold cross validation;", sample_per_fold, "samples per fold.")

        # sample all the rows at random (shuffling the dataset)
        X = data_labeled.sample(frac=1.0, random_state=42) 

        columns = ["Fold - "+str(i+1) for i in range(fold_count)]

        accuracies = pd.DataFrame.from_dict({
                'SC_test' : np.zeros(fold_count),
                'SC_train' : np.zeros(fold_count),
                'HW_test' : np.zeros(fold_count),
                'HW_train' : np.zeros(fold_count),
                'SW_test' : np.zeros(fold_count),
                'SW_train' : np.zeros(fold_count)
            },
        orient='index', columns=columns)

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
        
        # include averages
        accuracies.loc[:,'AVG'] = accuracies.mean(axis=1)

        # done, return the dataframe
        return accuracies

    # Label all the unlabeled samples using the model generated by
    # a fixed number of random samples
    def upsample(self, data_labeled, data_unlabeled, sample_count, addr_newlabels=None, online_weight_adjust=False, online_confidence_constant=0.25):

        print("Labeling", data_unlabeled.shape[0], "samples with information from", sample_count, "samples.")

        # sample the fixed number of samples from the labeled dataset at random
        data_labeled_subset = data_labeled.sample(n=sample_count, random_state=42)

        # default to this location
        if addr_newlabels == None:
            addr_newlabels = '../data/data-ss-labeled_'+str(sample_count)+'.csv'

        print("\t Calculating weights...", end='')
        weighted_dict = self.get_weighted_dict(data_labeled_subset)
        print("Done.")

        updates_made = 0

        # loop through each unlabeled sample and label them
        for i, data in data_unlabeled.iterrows():

            # get the sentance
            sentence = data.loc['MANUFACTURER_RECALL_REASON']

            # calculate the weight for this sample
            sample_label = self.weight_sentance(sentence, weighted_dict)

            # label the sample based on the weight from the model
            if sample_label['SC'] < 0.5:
                data_unlabeled.loc[i, 'SC'] = 0
            else:
                data_unlabeled.loc[i, 'SC'] = 1
            if sample_label['HW'] < 0.5:
                data_unlabeled.loc[i, 'HW'] = 0
            else:
                data_unlabeled.loc[i, 'HW'] = 1
            if sample_label['SW'] < 0.5:
                data_unlabeled.loc[i, 'SW'] = 0
            else:
                data_unlabeled.loc[i, 'SW'] = 1

            # Only adjust on confident weights
            if ((sample_label['SC'] < online_confidence_constant or 1-online_confidence_constant < sample_label['SC']) and 
                (sample_label['HW'] < online_confidence_constant or 1-online_confidence_constant < sample_label['HW']) and
                (sample_label['SW'] < online_confidence_constant or 1-online_confidence_constant < sample_label['SW']) and 
                online_weight_adjust
                ):

                weighted_dict = self.update_dict(weighted_dict, sentence, data_unlabeled.loc[i, 'SC'], data_unlabeled.loc[i, 'HW'], data_unlabeled.loc[i, 'SW'])
                updates_made += 1

        print('Updates made: ', updates_made)

        # write the now labeled data to the local csv
        data_unlabeled.to_csv(addr_newlabels, index=None)

        # return the labeled data
        return data_unlabeled

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

