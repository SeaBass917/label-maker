# ---------------------------------
# 
# ---------------------------------
from Util import tokenize

import pandas as pd
import numpy as np

class Grep():

    def __init__(self):

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

                                        # classify using the grep approach
    def clf_sentance(self, sentance, weighted_dict_SC, weighted_dict_HW, weighted_dict_SW):

        # tokenize the sentance
        words = tokenize(sentance)

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

    # for all top 1-100 weighted_dict evaluate the grep approach
    def sweep(self):

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
