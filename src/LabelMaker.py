# ------------------------------------------------------------
# S e a B a s s 
# 2 0 1 9
# 
# LabelMaker.py
# 
# Description - This GUI allows users to label the FDA recall 
# database while updating a weighted dictionary. This weighted
# dictionary will attempt to classify presented samples online
# allowing for real time observation of the performance.
# ------------------------------------------------------------
from WeightedDictionaryClf import WD

import tkinter as tk
import pandas as pd
import numpy as np

class Labeler():

    # initialize the vocabulary with the address of the corpus we will be labeling
    def __init__(self, 
                addr_labeled_data='data/recall_labeled.csv', 
                addr_unlabeled_data='data/recall_unlabeled.csv', 
                addr_weights='data/weighted-dictionary.pk1'):

        # location in filesystem for the datasets
        self.addr_labeled_data = addr_labeled_data
        self.addr_unlabeled_data = addr_unlabeled_data
        self.addr_weights = addr_weights

        # load the local data files
        self.load_data()

        # initialize the weighted dictionary classifier
        self.weighted_dict = WD(addr_weights=addr_weights)

        # load the weights from local storage,
        # if that fails regenerate them with our labeled data
        loadSuccess = self.weighted_dict.load_weights()
        if not loadSuccess:
            self.weighted_dict.refresh_weights(self.data_labeled)
            self.weighted_dict.save_weights()

        # init vars used for real time perfomance analysis
        self.samples_labeled_this_run = 0
        self.SC_correct = 0
        self.HW_correct = 0
        self.SW_correct = 0

    # technically its a method
    def dont_call_this_function(self):
        missing_the_magic_word = True
        while(missing_the_magic_word):
            print('ah ah ah, you didn\'t say the magic word')

    # save the labeled and unlabeled data to local storage
    def save_data(self):
        self.data_labeled.to_csv(self.addr_labeled_data, index=None)
        self.data_unlabeled.to_csv(self.addr_unlabeled_data, index=None)

    # read the labeled and unlabeled data from local storage
    def load_data(self):
        self.data_labeled = pd.read_csv(self.addr_labeled_data)
        self.data_unlabeled = pd.read_csv(self.addr_unlabeled_data)

    # find the index of the sample whose weight is closest to the weight specified
    def search_weight(self, weight=1.0):

        # initialize min weight difference and idx
        weight_diff_min = 1
        idx_max = 0
        
        # loop through a random subset of the dataset
        for _ in range(min(512, self.data_unlabeled.shape[0])):
            
            # grab sample at random
            i = int(np.random.uniform(0, self.data_unlabeled.shape[0]))
                
            # get sample -> calculate weight
            sample_label = self.weighted_dict.weight_sentance(self.data_unlabeled.iloc[i].loc['MANUFACTURER_RECALL_REASON'], self.weighted_dict)

            # update weight max
            weight_diff = abs(weight - sample_label[self.search_label.get()])
            if(weight_diff < weight_diff_min):
                weight_cur = sample_label
                weight_diff_min = weight_diff
                idx_max = i

        return idx_max, weight_cur
    
    # get a sample at random
    def get_sample(self):

        # grab sample at random
        i = int(np.random.uniform(0, self.data_unlabeled.shape[0]))
                
        # get sample -> calculate weight
        sample_label = self.weighted_dict.weight_sentance(self.data_unlabeled.loc[i,'MANUFACTURER_RECALL_REASON'], self.weighted_dict)

        return (i, sample_label)

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
        
        self.SC_perf['text'] = str(float(self.SC_correct) / float(self.samples_labeled_this_run))[:6]
        self.HW_perf['text'] = str(float(self.HW_correct) / float(self.samples_labeled_this_run))[:6]
        self.SW_perf['text'] = str(float(self.SW_correct) / float(self.samples_labeled_this_run))[:6]

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

        # update the weights based on the new label
        self.weighted_dict.update_model(self.sentance, y_SC=self.var_sc.get(), y_HW=self.var_hw.get(), y_SW=self.var_sw.get())

        # get next sample
        if(self.isUncertain.get()):
            self.next(weight=0.5)
        else:
            self.next()

    # search for a sample with the classification closest to 'weight'
    def next(self, weight=np.nan):

        # find next sample
        if np.isnan(weight):
            self.idx_cur, self.weight_cur = self.get_sample()
        else:
            self.idx_cur, self.weight_cur = self.search_weight(weight=weight)

        # update the model state for the user
        self.sample_label['text'] = 'Sample: ' + str(self.idx_cur)[:7]
        self.sample_count_label['text'] = 'Samples labeled: ' + str(self.weighted_dict['N_TOT']) + "(" + str(self.samples_labeled_this_run) + ")"
        self.SC_count_label['text'] = "Security: " + str(self.weighted_dict['N_SC']) 
        self.HW_count_label['text'] = "Hardware: " + str(self.weighted_dict['N_HW'])
        self.SW_count_label['text'] = "Software: " + str(self.weighted_dict['N_SW'])
        self.SC_label['text'] = ': ' + str(self.weight_cur['SC'])[:6]
        self.HW_label['text'] = ': ' + str(self.weight_cur['HW'])[:6]
        self.SW_label['text'] = ': ' + str(self.weight_cur['SW'])[:6]

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
        
        self.weight_cur = self.weighted_dict.weight_sentance(self.sentance, self.weighted_dict)

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

    # start running the program
    def run(self):
        
        ## initialize a gui for this
        
        # Greys   ["#ffffff","#f2f2f2","#e2e2e2","#cecece","#b4b4b4","#979797","#7a7a7a","#5f5f5f","#404040","#1e1e1e","#000000"]
        # Purples ["#f2f0f7","#dadaeb","#bcbddc","#9e9ac8","#807dba","#6a51a3","#4a1486"]
        # Oranges ["#feedde","#fdd0a2","#fdae6b","#fd8d3c","#f16913","#d94801","#8c2d04"]
        # Greens  ["#edf8e9","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#005a32"]
        # Blues   ["#eff3ff","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#084594"]

        # main window
        main_window = tk.Tk()
        main_window.title('Label Maker v4.0')
        main_window.configure(bg='#252525', padx=3, pady=3)
        
        # let the user decide what class we want to focus on
        self.isUncertain = tk.IntVar()
        tk.Checkbutton(main_window, bg='#252525', fg='#ffffff', highlightthickness=0, selectcolor="#404040", text='Find Uncertain', variable=self.isUncertain).grid(row=1, rowspan=2, column=2)
        self.search_label = tk.StringVar()
        self.search_label.set('SC')
        drop_down = tk.OptionMenu(main_window, self.search_label, *{'SC', 'HW', 'SW'})
        drop_down.configure(height=1, bg='#252525', fg='#ffffff', highlightthickness=0)
        drop_down['menu'].configure(bg='#252525', fg='#ffffff')
        drop_down.grid(row=1, rowspan=2, column=3, padx=1)
        
        # label to say the current sample number (like an id)
        self.sample_label = tk.Label(main_window, bg='#252525', fg='#ffffff')
        self.sample_label.grid(row=1, column=0, sticky=tk.W, padx=3, pady=3)

        # Idsplay number of abeled samples, and specify each classes frequency
        self.sample_count_label = tk.Label(main_window, bg='#252525', fg='#ffffff')
        self.SC_count_label = tk.Label(main_window, bg='#6a51a3', fg='#ffffff', width=13)
        self.HW_count_label = tk.Label(main_window, bg='#d94801', fg='#ffffff', width=13)
        self.SW_count_label = tk.Label(main_window, bg='#238b45', fg='#ffffff', width=13)
        self.sample_count_label.grid(row=2, column=0, sticky=tk.W, padx=3, pady=3)
        self.SC_count_label.grid(row=3, column=0, sticky=tk.W, padx=3, pady=3)
        self.HW_count_label.grid(row=4, column=0, sticky=tk.W, padx=3, pady=3)
        self.SW_count_label.grid(row=5, column=0, sticky=tk.W, padx=3, pady=3)
        
        # display a real time performance calc
        self.SC_perf = tk.Label(main_window, text="--", bg='#6a51a3', fg='#ffffff', width=7)
        self.HW_perf = tk.Label(main_window, text="--", bg='#d94801', fg='#ffffff', width=7)
        self.SW_perf = tk.Label(main_window, text="--", bg='#238b45', fg='#ffffff', width=7)
        self.SC_perf.grid(row=3, column=1, sticky=tk.W, padx=3, pady=3)
        self.HW_perf.grid(row=4, column=1, sticky=tk.W, padx=3, pady=3)
        self.SW_perf.grid(row=5, column=1, sticky=tk.W, padx=3, pady=3)

        # check boxes for voting
        self.var_sc = tk.IntVar()
        self.var_hw = tk.IntVar()
        self.var_sw = tk.IntVar()
        tk.Checkbutton(main_window, bg='#6a51a3', fg='#ffffff', highlightthickness=0, selectcolor="#4a1486", text='Security', variable=self.var_sc).grid(row=3, column=2, sticky=tk.W, padx=3, pady=3)
        tk.Checkbutton(main_window, bg='#d94801', fg='#ffffff', highlightthickness=0, selectcolor="#8c2d04", text='Hardware', variable=self.var_hw).grid(row=4, column=2, sticky=tk.W, padx=3, pady=3)
        tk.Checkbutton(main_window, bg='#238b45', fg='#ffffff', highlightthickness=0, selectcolor="#005a32", text='Software', variable=self.var_sw).grid(row=5, column=2, sticky=tk.W, padx=3, pady=3)

        # labels for the checkboxes
        self.SC_label = tk.Label(main_window, bg='#6a51a3', fg='#ffffff', highlightthickness=0, width=7)
        self.HW_label = tk.Label(main_window, bg='#d94801', fg='#ffffff', highlightthickness=0, width=7)
        self.SW_label = tk.Label(main_window, bg='#238b45', fg='#ffffff', highlightthickness=0, width=7)
        self.SC_label.grid(row=3, column=3, sticky=tk.W, padx=3, pady=3)
        self.HW_label.grid(row=4, column=3, sticky=tk.W, padx=3, pady=3)
        self.SW_label.grid(row=5, column=3, sticky=tk.W, padx=3, pady=3)

        # this is where the text will be pasted
        self.text_box = tk.Text(main_window, bg='#1e1e1e', fg='#ffffff', height=7, width=70)
        self.text_box.grid(row=7, columnspan=4, padx=3, pady=3)

        # skip button
        tk.Button(main_window, bg='#404040', fg='#ffffff', highlightthickness=0, text='Skip -->', width=15, bd=3, command=self.next).grid(row=8, column=3, padx=3, pady=3)

        # submit the label and get the next sample
        tk.Button(main_window, bg='#404040', fg='#ffffff', highlightthickness=0, text='Submit', height=2, width=15, bd=3, command=self.submit).grid(row=9, column=3, padx=3, pady=3)

        # Allow user to search for weighted_dict in unlabeled samples
        self.search_key = tk.StringVar()
        self.search_key.set("Battery...Software...")
        tk.Entry(main_window, textvariable=self.search_key, bg='#000000', fg='#ffffff', highlightthickness=0, width=15).grid(row=9, column=0, padx=3, pady=3)
        tk.Button(main_window, bg='#404040', fg='#ffffff', highlightthickness=0, text='Keyword Search', width=15, command=self.search_keyword, bd=3).grid(row=9, column=1, padx=3, pady=3)

        # call now to load the next(first) sample
        self.next()

        # put it up
        main_window.mainloop()
