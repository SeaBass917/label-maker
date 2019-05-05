import pandas as pd
import numpy as np 
import pickle as pk
from enum import Enum
import string
import tkinter as tk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

# weights are of the form (f_SECURITY, f_HW, f_SW, f_total)
# frequency as an integer, not ratio, hense the f_total
# 

# for indexing the labels
class Classes(Enum):
    Security = 0
    Hardware = 1
    Software = 2

class Labeler():

    # initialize the vocabulary with the address of the corpus we will be labeling
    def __init__(self, 
                addr_data='../data/recall_labeled.csv', 
                addr_weights='../data/recall_labeled-weights.pk1'):

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
        self.N_tot = 0
        for i, data in self.df.iterrows():
            if not np.isnan(data.iloc[-1]):
                self.N_tot += 1
                self.N_SC += data.loc['SECURITY']
                self.N_HW += data.loc['HW']
                self.N_SW += data.loc['SW']

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
                        self.keywords[word] = np.array([0,0,0,0])

                    self.keywords[word][0] += data.loc['SECURITY']
                    self.keywords[word][1] += data.loc['HW']
                    self.keywords[word][2] += data.loc['SW']
                    self.keywords[word][3] += 1
        
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
                weight = np.array([0,0,0,0])

            # add the new labels to the prev total
            # note the labels are taken from the gui checkbox values which are 0 or 1
            weight[0] += self.var_sc.get()
            weight[1] += self.var_hw.get()
            weight[2] += self.var_sw.get()
            weight[3] += 1

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
        # (close to ~0.5)
        norm_SC = 0.0
        norm_HW = 0.0
        norm_SW = 0.0

        # loop through the words in the recall
        for word in words:

            # look up the freuencies in the table
            weights = self.keywords.get(word)

            # if we arent in the table set to all zeros
            if weights is not None:
                
                weights = np.array(weights, dtype='float32')

                weights[0] /= weights[3]
                weights[1] /= weights[3]
                weights[2] /= weights[3]

                #try doubling weights that are near 0 or 1

                # we arent going to count weights close to 0.5
                # near 0.5 tells us the word doesnt really say 
                # much about the sentance
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

    def dont_call_this_function(self):
        missing_the_magic_word = True
        while(missing_the_magic_word):
            print('ah ah ah, you didn\'t say the magic word')

    def save_weights(self):
        with open(self.addr_weights, 'wb') as f:
            pk.dump(self.keywords, f, pk.HIGHEST_PROTOCOL)

    def load_weights(self):
        try:
            with open(self.addr_weights, 'rb') as f:
                self.keywords = pk.load(f)
        except:
            self.keywords = {}

    def save_data(self):
        self.df.to_csv(self.addr_data, index=None)

    def load_data(self):
        self.df = pd.read_csv(self.addr_data)

    # used to convert sentance to list of words
    # this removes stop words, punctuation, and numbers
    def tokenize(self, sentance):
        # tokenize the words, without punctuation
        words_ugly = word_tokenize(sentance.translate(dict((ord(char), None) for char in string.punctuation)), language='english')

        # remove numbers
        words_nonum = []
        for word in words_ugly:
            if not any(char.isdigit() for char in word):
                words_nonum.append(word)
        del words_ugly

        # load the stop words (a, the, and...)
        stop_words = set(stopwords.words('english'))

        # remove stop words and miss-spelled words
        words = []
        for word in words_nonum:
            if word.lower() not in stop_words:
                words.append(word.lower())

        return words

    # find the index of the sample we 
    # are most confident about
    # The L2 norm will tell us which sample we are absoultely sure of one way or another
    # a sum will specify that we are confident it is the classification 
    # of the weighted label
    # LIMIT TO SEARCHING SMALLER CHUNKS
    def search_weight(self, weight=1.0):

        # initialize min weight difference and idx
        weight_diff_min = 1
        idx_max = 0

        # check the dropdown box to see what class were looking for
        trait = Classes[self.search_label.get()].value
        
        # loop through each of the elements
        for r in range(512):
            
            # grab sample a random
            i = int(np.random.uniform(0, self.df.shape[0]))

            # if the last element is NaN then the data is unlabeled
            if np.isnan(self.df.iloc[i,-1]):
                
                # calculate weight for this sample
                sample_label = self.sample_weight(self.tokenize(self.df.loc[i,'MANUFACTURER_RECALL_REASON']))

                # update weight max
                weight_diff = abs(weight - sample_label[trait])
                if(weight_diff < weight_diff_min):
                    weight_max = sample_label
                    weight_diff_min = weight_diff
                    idx_max = i

        return idx_max, weight_max
    
    def submit_confident(self):

        # label the dataframe
        self.df.iloc[self.max_idx,-3] = self.var_sc.get()
        self.df.iloc[self.max_idx,-2] = self.var_hw.get()
        self.df.iloc[self.max_idx,-1] = self.var_sw.get()

        # update how many labeled sampled there are
        self.N_SC += self.var_sc.get()
        self.N_HW += self.var_hw.get()
        self.N_SW += self.var_sw.get()
        self.N_tot += 1

        # save the labels to local storage
        self.save_data()

        # tokenize the current sentance were looking at
        words = self.tokenize(self.sentance)

        # update the weights based on the new label
        self.weight_update(words)

        self.next(weight=1.0)
    
    def submit_unconfident(self):

        # label the dataframe
        self.df.iloc[self.max_idx,-3] = self.var_sc.get()
        self.df.iloc[self.max_idx,-2] = self.var_hw.get()
        self.df.iloc[self.max_idx,-1] = self.var_sw.get()

        # update how many labeled sampled there are
        self.N_SC += self.var_sc.get()
        self.N_HW += self.var_hw.get()
        self.N_SW += self.var_sw.get()
        self.N_tot += 1

        # save the labels to local storage
        self.save_data()

        # tokenize the current sentance were looking at
        words = self.tokenize(self.sentance)

        # update the weights based on the new label
        self.weight_update(words)

        self.next(weight=0.5)

    def next(self, weight=1.0):

        # get unlabeled sample with highest confidence
        self.max_idx, weight_max = self.search_weight(weight=weight)
        self.sample_label['text'] = 'Sample: ' + str(self.max_idx)[:7]
        self.sample_count_label['text'] = 'Samples labeled: ' + str(self.N_tot) + "\nSecurity: " + str(self.N_SC) + "\nHardware: " + str(self.N_HW) + "\nSoftware: " + str(self.N_SW)
        self.SC_label['text'] = ': ' + str(weight_max[0])
        self.HW_label['text'] = ': ' + str(weight_max[1])
        self.SW_label['text'] = ': ' + str(weight_max[2])

        # get the highest confidence sentance
        self.sentance = self.df.loc[self.max_idx,'MANUFACTURER_RECALL_REASON']

        # post it in the box
        self.text_box.delete('1.0', tk.END)
        self.text_box.insert(tk.END, self.sentance)

        # uncheck the checkboxes
        self.var_sc.set(0)
        self.var_hw.set(0)
        self.var_sw.set(0)

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
        self.sample_count_label['text'] = 'Samples labeled: ' + str(self.N_tot) + "\nSecurity: " + str(self.N_SC) + "\nHardware: " + str(self.N_HW) + "\nSoftware: " + str(self.N_SW)
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

    # test the performance of the model based on the labeled samples
    def test(self):

        print("Evaluating performance...")

        SC_acc = 0.0
        SW_acc = 0.0
        HW_acc = 0.0
        labeled_count = 0

        # for each labeled sample
        for i, data in self.df.iterrows():
            if not np.isnan(data.iloc[-1]):

                # increment number of total labeled samples
                labeled_count += 1

                # calculate weight for this sample
                sample_label = self.sample_weight(self.tokenize(self.df.loc[i,'MANUFACTURER_RECALL_REASON']))

                # if the model is correct increment the the respective counter
                if(abs(self.df.loc[i,'SECURITY'] - sample_label[0]) < 0.5):
                    SC_acc += 1
                if(abs(self.df.loc[i,'HW'] - sample_label[1]) < 0.5):
                    HW_acc += 1
                if(abs(self.df.loc[i,'SW'] - sample_label[2]) < 0.5):
                    SW_acc += 1
        
        SC_acc /= labeled_count
        HW_acc /= labeled_count
        SW_acc /= labeled_count

        print("Accuracy for Security Threats:", SC_acc)
        print("Accuracy for Hardware Issues:", HW_acc)
        print("Accuracy for Software Issues:", SW_acc)

    # start running the program
    def run(self):
        
        ## initialize a gui for this

        # main window
        main_window = tk.Tk()
        main_window.title('Label Maker v2.1')
        main_window.configure(bg='#404040')
 
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='>', height=20, command=self.next).grid(rowspan=6, column=4)
        
        # label to say the current sample number and how the weights feel about the sample
        self.sample_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.sample_count_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.SC_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.HW_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.SW_label = tk.Label(main_window, bg='#404040', fg='#ffffff')
        self.sample_label.grid(row=0, column=0, sticky=tk.W)
        self.sample_count_label.grid(row=1, column=0, sticky=tk.W)
        self.SC_label.grid(row=2, column=1, sticky=tk.W)
        self.HW_label.grid(row=3, column=1, sticky=tk.W)
        self.SW_label.grid(row=4, column=1, sticky=tk.W)

        # this is where the text will be pasted
        self.text_box = tk.Text(main_window, bg='#202020', fg='#ffffff', width=50, height=10)
        self.text_box.grid(row=5, columnspan=3)

        # check boxes for voting
        self.var_sc = tk.IntVar()
        self.var_hw = tk.IntVar()
        self.var_sw = tk.IntVar()
        tk.Checkbutton(main_window, bg='#404040', fg='#ffffff', selectcolor="#202020", text='Security', variable=self.var_sc).grid(row=2, column=0, sticky=tk.W)
        tk.Checkbutton(main_window, bg='#404040', fg='#ffffff', selectcolor="#202020", text='Hardware', variable=self.var_hw).grid(row=3, column=0, sticky=tk.W)
        tk.Checkbutton(main_window, bg='#404040', fg='#ffffff', selectcolor="#202020", text='Software', variable=self.var_sw).grid(row=4, column=0, sticky=tk.W)

        self.search_label = tk.StringVar()
        self.search_label.set('Security')

        # Change what class we want to focus on
        drop_down = tk.OptionMenu(main_window, self.search_label, *{'Security', 'Hardware', 'Software'})
        drop_down.configure(bg='#404040', fg='#ffffff')
        drop_down['menu'].configure(bg='#404040', fg='#ffffff')
        drop_down.grid(row=6, column=0)
        
        # submit the vote/label and get the next sample
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='Submit\nGet Confident Sample', width=25, command=self.submit_confident).grid(row=6, column=1)
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='Submit\nGet Unconfident Sample', width=25, command=self.submit_unconfident).grid(row=6, column=2)

        self.search_key = tk.StringVar()
        self.search_key.set("Battery...Software...Problem...")
        tk.Label(main_window, text="Keyword Search", bg='#404040', fg='#ffffff').grid(row=7, column=0)
        tk.Entry(main_window, textvariable=self.search_key, bg='#202020', fg='#ffffff').grid(row=7, column=1)
        tk.Button(main_window, bg='#404040', fg='#ffffff', text='Search', width=25, command=self.search_keyword).grid(row=7, column=2)

        # call now to load the next(first) sample
        self.next()

        # put it up
        main_window.mainloop()

l = Labeler()
l.test()

