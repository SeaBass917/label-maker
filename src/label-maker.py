import tkinter as tk
import pandas as pd
import numpy as np 
import pickle as pk
from enum import Enum
import string
import nltk
nltk.download('stopwords') # stopwords 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

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
                addr_data='../data/recall_labeled.csv', 
                addr_weights='../data/weighted-dictionary.pk1'):

        # location in filesystem for the datasets
        self.addr_data = addr_data
        self.addr_weights = addr_weights

        # load the local data files
        self.load_data()
        self.load_weights()

        # read the dataset and count how many 
        # samples have labels on them
        self.N_SC = 0
        self.N_HW = 0
        self.N_SW = 0
        self.N_SWHW = 0
        self.N_TOT = 0
        for i, data in self.df.iterrows():
            if not np.isnan(data.iloc[-1]):
                self.N_TOT += 1
                self.N_SC += data.loc['SC']
                self.N_HW += data.loc['HW']
                self.N_SW += data.loc['SW']
                if(data.loc['SW'] == 1 and data.loc['HW'] == 1):
                    self.N_SWHW += 1
        
        # init vars used for real time perfomance analysis
        self.samples_labeled_this_run = 0
        self.SC_correct = 0
        self.HW_correct = 0
        self.SW_correct = 0

    # for when I edit the dataset manually
    def refresh_weights(self):
        
        # new keywords dict
        self.keywords = {}

        # loop through dataset, ignore unlabeled rows
        # recount the frequencies for the keywords
        for i, data in self.df.iterrows():
            if not np.isnan(data.iloc[-1]):

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
            self.keywords = {}

    # save the labels to a local file
    def save_data(self):
        self.df.to_csv(self.addr_data, index=None)

    # read the labels from a local file
    def load_data(self):
        self.df = pd.read_csv(self.addr_data)

    # used to convert sentance to list of words
    # this removes stop words, punctuation, and numbers
    def tokenize(self, sentance):

        # tokenize the words, without punctuation
        # TODO wtf did I write here, this can't be the best way to do that
        words_ugly = word_tokenize(sentance.translate(dict((ord(char), None) for char in string.punctuation)), language='english')

        # remove numbers
        words_nonum = []
        for word in words_ugly:
            if not any(char.isdigit() for char in word):
                words_nonum.append(word)
        del words_ugly

        # load the stop words from nltk (a, the, and...)
        stop_words = set(stopwords.words('english'))

        # remove stop words and miss-spelled words
        words = []
        for word in words_nonum:
            if word.lower() not in stop_words:
                words.append(word.lower())

        return words

    # find the index of the sample whose weight is closest to the weight specified
    def search_weight(self, weight=1.0):

        # initialize min weight difference and idx
        weight_diff_min = 1
        idx_max = 0

        # check the dropdown box to see what class were looking for
        trait = Classes[self.search_label.get()].value
        
        # loop through a random subset of the dataset
        for r in range(512):
            
            # grab sample a random
            i = int(np.random.uniform(0, self.df.shape[0]))

            # if the last element is NaN then the data is unlabeled
            if np.isnan(self.df.iloc[i,-1]):
                
                # get sample -> tokenize -> calculate weight
                sample_label = self.sample_weight(self.tokenize(self.df.loc[i,'MANUFACTURER_RECALL_REASON']))

                # update weight max
                weight_diff = abs(weight - sample_label[trait])
                if(weight_diff < weight_diff_min):
                    weight_max = sample_label
                    weight_diff_min = weight_diff
                    idx_max = i

        return idx_max, weight_max
    
    # commit a user specified label to the dictionary
    def submit(self):

        # calculate real time performance based on the submitted label
        # and the models prediction
        self.samples_labeled_this_run += 1
        if(abs(self.var_sc.get() - self.weight_max[0]) < 0.5):
            self.SC_correct += 1
        if(abs(self.var_hw.get() - self.weight_max[1]) < 0.5):
            self.HW_correct += 1
        if(abs(self.var_sw.get() - self.weight_max[2]) < 0.5):
            self.SW_correct += 1
        
        self.SC_perf['text'] = str(float(self.SC_correct) / float(self.samples_labeled_this_run))
        self.HW_perf['text'] = str(float(self.HW_correct) / float(self.samples_labeled_this_run))
        self.SW_perf['text'] = str(float(self.SW_correct) / float(self.samples_labeled_this_run))

        # label the dataframe
        self.df.iloc[self.max_idx,-3] = self.var_sc.get()
        self.df.iloc[self.max_idx,-2] = self.var_hw.get()
        self.df.iloc[self.max_idx,-1] = self.var_sw.get()
        
        # update how many labeled sampled there are
        self.N_SC += self.var_sc.get()
        self.N_HW += self.var_hw.get()
        self.N_SW += self.var_sw.get()
        self.N_TOT += 1

        # save the labels to local storage
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

        # find sample and
        # update the model state for the user
        self.max_idx, self.weight_max = self.search_weight(weight=weight)
        self.sample_label['text'] = 'Sample: ' + str(self.max_idx)[:7]
        self.sample_count_label['text'] = 'Samples labeled: ' + str(self.N_TOT) + "(" + str(self.samples_labeled_this_run) + ")"
        self.SC_count_label['text'] = "Security: " + str(self.N_SC) 
        self.HW_count_label['text'] = "Hardware: " + str(self.N_HW)
        self.SW_count_label['text'] = "Software: " + str(self.N_SW)
        self.SC_label['text'] = ': ' + str(self.weight_max[0])
        self.HW_label['text'] = ': ' + str(self.weight_max[1])
        self.SW_label['text'] = ': ' + str(self.weight_max[2])

        # paste that recall into the textbox
        self.sentance = self.df.loc[self.max_idx,'MANUFACTURER_RECALL_REASON']
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
        for i, data in self.df.iterrows():
            if np.isnan(data.iloc[-1]):
                self.sentance = data.loc['MANUFACTURER_RECALL_REASON']
                if keyword in self.sentance:
                    match = True
                    self.max_idx = i
                    break
        # if there is no match then set these
        if not match:
            i = -1
            self.sentance = "<No Keyword Match>"
        
        weight_max = self.sample_weight(self.tokenize(self.sentance))
        self.sample_label['text'] = 'Sample: ' + str(self.max_idx)[:7]
        self.sample_count_label['text'] = 'Samples labeled: ' + str(self.N_TOT) + "\nSecurity: " + str(self.N_SC) + "\nHardware: " + str(self.N_HW) + "\nSoftware: " + str(self.N_SW)
        self.SC_label['text'] = ': ' + str(weight_max[0])
        self.HW_label['text'] = ': ' + str(weight_max[1])
        self.SW_label['text'] = ': ' + str(weight_max[2])

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

        return (SC_top, HW_top, SW_top)

    # test the performance of the model based on the labeled samples
    def test(self):

        print("Evaluating performance...")

        # the grep keywords from alemzadeh's work
        keywords_SW = [ 'software', 'application', 'function', 'code', 
                        'version', 'backup', 'database', 'program', 
                        'bug', 'java', 'run', 'upgrade']
        keywords_HW = [ 'board', 'chip', 'hardware', 'processor', 
                        'memory', 'disk', 'PCB', 'electronic', 
                        'electrical', 'circuit', 'leak', 'short-circuit', 
                        'capacitor', 'transistor', 'resistor', 'battery', 
                        'power', 'supply', 'outlet', 'plug', 'power-up', 
                        'discharge', 'charger']

        # grep keywords using Alemzadeh's approach but generated with my top 15 keywords
        keywords_SC = ['one', 'system', 'patient', 'images', 'software', 'results', 
                        'image', 'another', 'data', 'potential', 'incorrect', 'study', 
                        'id', 'patients', 'may']

        # accurcies for the weighted dictionary and grep approaches
        SC_acc_WD = 0.0
        SW_acc_WD = 0.0
        HW_acc_WD = 0.0
        SC_acc_grep = 0.0
        SW_acc_grep = 0.0
        HW_acc_grep = 0.0
        labeled_count = 0

        # for each labeled sample
        for i, data in self.df.iterrows():
            if not np.isnan(data.iloc[-1]):

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
        "Security: ", self.N_SC, "\n",
        "Hardware: ", self.N_HW, "\n",
        "Software: ", self.N_SW, "\n",
        "HW&SW:    ", self.N_SWHW, "\n",
        "Other:    ", self.N_TOT - (self.N_SC + self.N_HW + self.N_SW - self.N_SWHW), "\n",
        "Total:    ", self.N_TOT)

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



l = Labeler()
#SC, HW, SW = l.top_keyword_get()
#print('Security')
#print(SC)
#print('Hardware')
#print(HW)
#print('Software')
#print(SW)
#l.test()
#l.summary()
l.run()