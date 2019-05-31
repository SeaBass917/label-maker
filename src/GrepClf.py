# ------------------------------------------------------------
# S e a B a s s 
# 2 0 1 9
# 
# GrepClf.py
# 
# Description - This classifier uses a list of keywords to 
# classify sentances.
# ------------------------------------------------------------
from Util import tokenize

import pandas as pd
import numpy as np

class Grep():

    def __init__(self, keywords_SC, keywords_HW, keywords_SW):

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

        # internally store the keywords given
        self.keywords_SC = keywords_SC
        self.keywords_HW = keywords_HW
        self.keywords_SW = keywords_SW

                                        # classify using the grep approach
    def clf_sentance(self, sentance):

        # tokenize the sentance
        words = tokenize(sentance)

        # initialize the classifications
        classifications = {'SC': 0, 'HW': 0, 'SW': 0}

        # for each class look for weighted_dict, 
        # flag a classification on a keyword match
        # NOTE: weighted_dict is a list of tuples
        for word in words:
            for kw in self.keywords_SC:
                if(word == kw[0]):
                    classifications['SC'] = 1
            for kw in self.keywords_HW:
                if(word == kw[0]):
                    classifications['HW'] = 1
            for kw in self.keywords_SW:
                if(word == kw[0]):
                    classifications['SW'] = 1

        return classifications