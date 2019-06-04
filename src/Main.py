# ------------------------------------------------------------
# S e a B a s s 
# 2 0 1 9
# 
# Main.py
# 
# Description - Main entry point.
# ------------------------------------------------------------
from WeightedDictionaryClf import WD
from LabelMaker import Labeler

import pandas as pd
import numpy as np

# quantitatively analyze how much the models predictions are changing 
# as more data is used to label the samples
def measure_stability(reupsample=False):

    # init weighted dict
    w = WD()
    w.load_weights()
    
    # load the recall data
    data_labeled = pd.read_csv('../data/recall_labeled.csv')
    data_unlabeled = pd.read_csv('../data/recall_unlabeled.csv')

    # keep track of current and previous dataframes
    data_ss_labeled_prev = None
    data_ss_labeled_curr = None 

    # every round add a hundred samples, but the first few should be smaller incremenets
    # this array represents that
    sample_counts = np.concatenate([[1, 5, 15, 25, 50, 75], np.arange(100, data_labeled.shape[0], 50)])

    # initialize frame to store stability info
    stability = pd.DataFrame(index=['SC', 'HW', 'SW'], columns=sample_counts)

    for sample_count in sample_counts:

        print(sample_count, "...")
        
        # set the current as the previous
        data_ss_labeled_prev = data_ss_labeled_curr

        # get the semi-supervised labeled data
        if(reupsample):
            data_ss_labeled_curr = w.upsample(data_labeled, data_unlabeled, sample_count)
        else:
            # try to read if the data is there, if its not regenerate it
            try:
                data_ss_labeled_curr = pd.read_csv('../data/data-ss-labeled_'+str(sample_count)+'.csv')
            except:
                data_ss_labeled_curr = w.upsample(data_labeled, data_unlabeled, sample_count)
            
        total_same_SC = 0
        total_same_HW = 0
        total_same_SW = 0
        
        # Check that both previous and current dataframes are valid
        # they wont be on the first round so just keep 0's in total same
        if(sample_count != sample_counts[0]):

            # loop through each sample index and compare the two
            for i in range(data_ss_labeled_curr.shape[0]):

                # The prediction is the same as last time then increment the respective counter
                if(data_ss_labeled_curr.loc[i, 'SC'] == data_ss_labeled_prev.loc[i, 'SC']):
                    total_same_SC += 1
                if(data_ss_labeled_curr.loc[i, 'HW'] == data_ss_labeled_prev.loc[i, 'HW']):
                    total_same_HW += 1
                if(data_ss_labeled_curr.loc[i, 'SW'] == data_ss_labeled_prev.loc[i, 'SW']):
                    total_same_SW += 1

        # store the stability information gathered
        stability.loc['SC', sample_count] = float(total_same_SC) / data_ss_labeled_curr.shape[0]
        stability.loc['HW', sample_count] = float(total_same_HW) / data_ss_labeled_curr.shape[0]
        stability.loc['SW', sample_count] = float(total_same_SW) / data_ss_labeled_curr.shape[0]

    # store the resultant info locally
    stability.to_csv('../data/stability.csv')

    # return the data frame
    return stability

def print_stats(data_labeled):

    N_SC = 0
    N_HW = 0
    N_SW = 0
    N_SCHW = 0
    N_SCSW = 0
    N_HWSW = 0
    N_SCHWSW = 0
    N_OTHER = 0
    N_TOT = data_labeled.shape[0]

    for _, data in data_labeled.iterrows():

        if data.loc['SC']:
            N_SC += 1
        if data.loc['HW']:
            N_HW += 1
        if data.loc['SW']:
            N_SW += 1
        if data.loc['SC'] and data.loc['HW']:
            N_SCHW += 1
            print("\tWarning SCHW:", data.loc['MANUFACTURER_RECALL_REASON'])
        if data.loc['SC'] and data.loc['SW']:
            N_SCSW += 1
        if data.loc['HW'] and data.loc['SW']:
            N_HWSW += 1
        if data.loc['SC'] and data.loc['HW'] and data.loc['SW']:
            N_SCHWSW += 1
        if not data.loc['SC'] and not data.loc['HW'] and not data.loc['SW']:
            N_OTHER += 1

    
    print(
        "Class Frequencies\n",
        "Security: ", N_SC,    "\n",
        "Hardware: ", N_HW,    "\n",
        "Software: ", N_SW,    "\n",
        "SC&HW:    ", N_SCHW,  "\n",
        "SC&SW:    ", N_SCSW,  "\n",
        "HW&SW:    ", N_HWSW,  "\n",
        "SC&HW&SW: ", N_SCHWSW,"\n",
        "Other:    ", N_OTHER, "\n",
        "Total:    ", N_TOT,   "\n"
    )

def main():

    #measure_stability()


    #data_labeled = pd.read_csv('../data/recall_labeled.csv')
    #data_ss_labeled = pd.read_csv('../data/data-ss-labeled_800.csv')

    #print("--- True Label ---")
    #print_stats(data_labeled)

    #print("--- Model Label ---")
    #print_stats(data_ss_labeled)

    l = Labeler()
    l.run()


main()
