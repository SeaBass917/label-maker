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
def measure_stability(reupsample=False, online_weight_adjust=True, online_confidence_constant=0.25):

    # init weighted dict
    w = WD()
    w.load_weights()
    
    # load the recall data
    data_labeled = pd.read_csv('../data/recall_labeled.csv')
    data_unlabeled = pd.read_csv('../data/recall_unlabeled.csv')

    # keep track of current and previous dataframes
    data_ss_labeled_prev = None
    data_ss_labeled_curr = pd.DataFrame() 

    # every round add a hundred samples, but the first few should be smaller incremenets
    # this array represents that
    sample_counts = np.arange(100, data_labeled.shape[0]+1, 100)

    # initialize frame to store stability info
    stability = pd.DataFrame(index=['SC', 'HW', 'SW'], columns=sample_counts)

    for sample_count in sample_counts:

        print(sample_count, "...")

        if(online_weight_adjust):
            addr_data = '../data/data-ss-labeled_'+str(sample_count)+'(ON-'+str(online_confidence_constant)+').csv'
        else:
            addr_data = '../data/data-ss-labeled_'+str(sample_count)+'(OFF).csv'
        
        # set the current as the previous
        data_ss_labeled_prev = data_ss_labeled_curr.copy()

        # get the semi-supervised labeled data
        if(reupsample):
            data_ss_labeled_curr = w.upsample(data_labeled, data_unlabeled, sample_count, addr_newlabels=addr_data, online_weight_adjust=online_weight_adjust, online_confidence_constant=online_confidence_constant)
        else:
            # try to read if the data is there, if its not regenerate it
            try:
                data_ss_labeled_curr = pd.read_csv(addr_data)
            except:
                data_ss_labeled_curr = w.upsample(data_labeled, data_unlabeled, sample_count, addr_newlabels=addr_data, online_weight_adjust=online_weight_adjust, online_confidence_constant=online_confidence_constant)
            
        total_same_SC = 0
        total_same_HW = 0
        total_same_SW = 0
        total_comp_samples = 0
        
        # Check that both previous and current dataframes are valid
        # they wont be on the first round so just keep 0's in total same
        if(sample_count != sample_counts[0]):

            # loop through each sample index and compare the two
            for i in range(data_ss_labeled_curr.shape[0]):

                # don't bother on samples that are not computer at all
                if(data_ss_labeled_curr.loc[i, 'SC'] != 0 or data_ss_labeled_curr.loc[i, 'HW'] != 0 or data_ss_labeled_curr.loc[i, 'SW'] != 0):
                    
                    # The prediction is the same as last time then increment the respective counter
                    if(data_ss_labeled_curr.loc[i, 'SC'] == data_ss_labeled_prev.loc[i, 'SC']):
                        total_same_SC += 1
                    if(data_ss_labeled_curr.loc[i, 'HW'] == data_ss_labeled_prev.loc[i, 'HW']):
                        total_same_HW += 1
                    if(data_ss_labeled_curr.loc[i, 'SW'] == data_ss_labeled_prev.loc[i, 'SW']):
                        total_same_SW += 1

                    total_comp_samples += 1

        # store the stability information gathered
        if(total_comp_samples > 0):
            stability.loc['SC', sample_count] = float(total_same_SC) / total_comp_samples
            stability.loc['HW', sample_count] = float(total_same_HW) / total_comp_samples
            stability.loc['SW', sample_count] = float(total_same_SW) / total_comp_samples
        else:
            stability.loc['SC', sample_count] = -3.1415926
            stability.loc['HW', sample_count] = -3.1415926
            stability.loc['SW', sample_count] = -3.1415926

    # store the resultant info locally
    if online_weight_adjust:
        fileNameOut = '../data/stability(ON-'+str(online_confidence_constant)+').csv'
    else:
        fileNameOut = '../data/stability(OFF).csv'
    
    stability.to_csv(fileNameOut)

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

# Find and display top keywords fopr the weights in local storage
def show_top_keywords():

    # init weighted dict
    w = WD()

    # load the weights
    w.load_weights()

    # Find and display top keywords
    print('\t --- Top 100 Keywords ---')
    w.top_keyword_get(verbose=True)
    print("\n\n")

    # find and dislay stopwords
    print("\t --- Stop words ---")
    w.get_stop_words(verbose=True)
    print("\n\n")

# Run cross validation on the labeled dataset using the WD model
# TODO: Compare the Grep approach at the end
def test_performance(alpha=0, fold_count=8, addr_out='../data/test_out.csv'):

    # init weighted dict
    w = WD(alpha=alpha)
    w.load_weights()
    
    # load the recall data
    data_labeled = pd.read_csv('../data/recall_labeled.csv')

    # run X val
    accuracies = w.test(data_labeled, fold_count)

    # write results to local CSV
    accuracies.to_csv(addr_out)

def sweep_alphas(alpha_min, alpha_max, alpha_delta, fold_count=8):

    print('Sweep sigmoid alphas in classification.')

    for alpha in range(alpha_min, alpha_max, alpha_delta):

        print('\t --- Alpha:', alpha)
        test_performance(alpha=alpha, addr_out='../data/'+str(fold_count)+'-fold_X-val_alpha-'+str(alpha)+'.csv')

def sweep_confidence_online_weight_adjust(alpha_min, alpha_max, alpha_delta):

    print('Sweeping online confidence constants...')
    alpha = alpha_min
    while(alpha <= alpha_max):
        print('\t --- Alpha:', alpha)
        measure_stability(reupsample=True, online_weight_adjust=True, online_confidence_constant=alpha)
        alpha += alpha_delta

def main():

    #measure_stability(reupsample=True, online_weight_adjust=True)
    measure_stability(reupsample=True, online_weight_adjust=False)

    sweep_confidence_online_weight_adjust(0.0, 0.15, 0.05)

    #show_top_keywords()

    #test_performance(alpha=40)

    #sweep_alphas(5, 40, 5)

    
    #Labeler().run()

    #data_labeled = pd.read_csv('../data/recall_labeled.csv')
    #data_ss_labeled = pd.read_csv('../data/data-ss-labeled_800.csv')

    #print("--- True Labels ---")
    #print_stats(data_labeled)

    #print("--- Model Labels ---")
    #print_stats(data_ss_labeled)

main()
