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
from Util import tokenize

import pandas as pd
import numpy as np

# quantitatively analyze how much the models predictions are changing 
# as more data is used to label the samples
def measure_stability(reupsample=False, ratio_noncomp_samples=0.5):
    
    # init weighted dict
    w = WD()
    w.load_weights()
    
    # load the recall data
    data_labeled = pd.read_csv('data/recall_labeled.csv')
    data_unlabeled = pd.read_csv('data/recall_unlabeled.csv')

    # Filter the data betwen computer and non-computer related issues
    data_labeled_comp = pd.DataFrame()
    data_labeled_noncomp = pd.DataFrame()
    for i, data in data_labeled.iterrows():
        if(data['SC'] == 1 or data['HW'] == 1 or data['SW'] == 1):
            data_labeled_comp = data_labeled_comp.append(data)
        else:
            data_labeled_noncomp = data_labeled_noncomp.append(data)

    print(data_labeled_comp.shape[0], 'computer related samples')
    print(data_labeled_noncomp.shape[0], 'non-computer related samples')

    # keep track of current and previous dataframes
    data_ss_labeled_prev = None
    data_ss_labeled_curr = pd.DataFrame() 

    # every round add a hundred samples, but the first few should be smaller incremenets
    # this array represents that
    sample_counts = np.arange(50, data_labeled_comp.shape[0]+1, 50)

    # initialize frame to store stability info
    stability = pd.DataFrame(index=['SC', 'HW', 'SW'], columns=sample_counts)

    for sample_count in sample_counts:

        print(sample_count, "...")

        addr_data = 'data/data-ss-labeled_'+str(sample_count)+'('+str(ratio_noncomp_samples)+').csv'
        
        # set the current as the previous
        data_ss_labeled_prev = data_ss_labeled_curr.copy()

        # get the semi-supervised labeled data
        if(reupsample):
            data_ss_labeled_curr = w.upsample(data_labeled_comp, data_labeled_noncomp, data_unlabeled, sample_count, addr_newlabels=addr_data, ratio_noncomp_samples=ratio_noncomp_samples)
        else:
            # try to read if the data is there, if its not regenerate it
            try:
                data_ss_labeled_curr = pd.read_csv(addr_data)
            except:
                data_ss_labeled_curr = w.upsample(data_labeled_comp, data_labeled_noncomp, data_unlabeled, sample_count, addr_newlabels=addr_data, ratio_noncomp_samples=ratio_noncomp_samples)
            
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
    fileNameOut = 'data/stability('+str(ratio_noncomp_samples)+').csv'
    
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
def test_performance(alpha=0, fold_count=8, sample_count=-1, ratio_noncomp_samples=0.5, addr_out='data/test_out.csv'):

    # init weighted dict
    w = WD(alpha=alpha)
    w.load_weights()
    
    # load the recall data
    data_labeled = pd.read_csv('data/recall_labeled.csv')

    # Filter the data betwen computer and non-computer related issues
    data_labeled_comp = pd.DataFrame()
    data_labeled_noncomp = pd.DataFrame()
    for i, data in data_labeled.iterrows():
        if(data['SC'] == 1 or data['HW'] == 1 or data['SW'] == 1):
            data_labeled_comp = data_labeled_comp.append(data)
        else:
            data_labeled_noncomp = data_labeled_noncomp.append(data)

    print(data_labeled_comp.shape[0], 'computer related samples')
    print(data_labeled_noncomp.shape[0], 'non-computer related samples')

    # determine how many non computer related samples to include
    sample_count_noncomp = min(int(sample_count * ratio_noncomp_samples), data_labeled_noncomp.shape[0])

    if(sample_count_noncomp == data_labeled_noncomp.shape[0]):
        print('\tWarning! Not enough non computer related samples to fill the ratio.')

    # sample the fixed number of samples from the labeled datasets at random
    data_labeled_subset = data_labeled_comp.sample(n=sample_count, random_state=42).append(
        data_labeled_noncomp.sample(n=sample_count_noncomp, random_state=42)
    )

    # run X val
    accuracies = w.test(data_labeled_subset, fold_count)

    # write results to local CSV
    accuracies.to_csv(addr_out)

# TODO: Compare the Grep approach at the end
def sweep_performance(ratio_noncomp_samples=0.5):
    for sample_count in range(50, 551, 50):
        test_performance(alpha=0, fold_count=8, sample_count=sample_count, ratio_noncomp_samples=ratio_noncomp_samples, 
                         addr_out='data/performance_sweeps/test_'+str(sample_count)+'('+str(ratio_noncomp_samples)+').csv')

def sweep_alphas(alpha_min, alpha_max, alpha_delta, fold_count=8):

    print('Sweep sigmoid alphas in classification.')

    for alpha in range(alpha_min, alpha_max, alpha_delta):

        print('\t --- Alpha:', alpha)
        test_performance(alpha=alpha, addr_out='data/'+str(fold_count)+'-fold_X-val_alpha-'+str(alpha)+'.csv')

def sweep_confidence_online_weight_adjust(alpha_min, alpha_max, alpha_delta):

    print('Sweeping online confidence constants...')
    alpha = alpha_min
    while(alpha <= alpha_max):
        print('\t --- Alpha:', alpha)
        measure_stability(reupsample=True, online_weight_adjust=True, online_confidence_constant=alpha)
        alpha += alpha_delta

def classify_samples(data_unlabeled_addr='data/recall_unlabeled.csv'):
    
    model = WD()
    model.load_weights()

    data_unlabeled = pd.read_csv(data_unlabeled_addr)

    for i, row in data_unlabeled.iterrows():
        
        sentance = row.loc['MANUFACTURER_RECALL_REASON']

        label = model.clf_WD(sentance)

        data_unlabeled['SC'][i] = label['SC']
        data_unlabeled['HW'][i] = label['HW']
        data_unlabeled['SW'][i] = label['SW']

    data_unlabeled.to_csv('data/recall_labeled_from_model.csv')

def term_freq_analysis(data_labeled_from_model_addr='data/recall_labeled_from_model.csv',
                        data_labeled_addr='data/recall_labeled.csv'):

    software_kwoi = [
        ['anomoly'],
        ['image', 'imaging'],
        ['interface', 'gui'],
        ['version', 'v'],
        ['protocol', 'message']
    ]
    hardware_kwoi = [
        ['defective', 'damaged'],
        ['battery', 'power', 'charge', 'energy', 'voltage', 'charging', 'charger'],
        ['board', 'circuit', 'capacitor', 'wiring', 'pcb'],
        ['alarm'],
        ['monitor', 'display']
    ]
    security_kwoi = [
        ['error'],
        ['sent', 'transfer', 'recieved'],
        ['deleted', 'corrupted'],
        ['anomaly'],
        ['data', 'file', 'information', 'disk', 'archive', 'record']
    ]

    # read both the samples labeled
    df_man = pd.read_csv(data_labeled_addr)
    df_auto = pd.read_csv(data_labeled_from_model_addr)

    # combine them
    data = pd.concat([df_man, df_auto], ignore_index=True)

    # determine number of years
    # it should be 17, from 2002 to 2018
    years = data['YEAR']
    years = years.drop_duplicates()
    numYears = years.shape[0]

    # seperate hist for each class
    # determing most popular words for each year
    # store in dict with year key
    hist_global_sc = {}
    hist_global_sw = {}
    hist_global_hw = {}
    hist_by_year_sc = {}
    hist_by_year_sw = {}
    hist_by_year_hw = {}
    for year in years:
        hist_by_year_sc[year] = {}
        hist_by_year_sw[year] = {}
        hist_by_year_hw[year] = {}

    # go through each sample
    # update hist by year and class
    for i, row in data.iterrows():

        # read the row
        year = row['YEAR']
        sentance = row['MANUFACTURER_RECALL_REASON']
        label_SC = row['SC']
        label_SW = row['SW']
        label_HW = row['HW']

        words = tokenize(sentance)

        # for each class update the hist
        if label_SC == 1:
            hist = hist_by_year_sc[year]
            
            for word in words:
                if word in hist.keys():
                    hist[word] += 1
                else:
                    hist[word] = 1
                
                if word in hist_global_sc:
                    hist_global_sc[word] += 1
                else:
                    hist_global_sc[word] = 1

            hist_by_year_sc[year] = hist
            
        if label_SW == 1:
            hist = hist_by_year_sw[year]
            
            for word in words:
                if word in hist.keys():
                    hist[word] += 1
                else:
                    hist[word] = 1
                
                if word in hist_global_sw:
                    hist_global_sw[word] += 1
                else:
                    hist_global_sw[word] = 1

            hist_by_year_sw[year] = hist
            
        if label_HW == 1:
            hist = hist_by_year_hw[year]
            
            for word in words:
                if word in hist.keys():
                    hist[word] += 1
                else:
                    hist[word] = 1
                
                if word in hist_global_hw:
                    hist_global_hw[word] += 1
                else:
                    hist_global_hw[word] = 1

            hist_by_year_hw[year] = hist
            
    # dataframes to be filled for each class and output of this function
    df_sc = pd.DataFrame({
        'words': [0 for year in years]
    }, index=years)

    df_sw = pd.DataFrame({
        'words': [0 for year in years]
    }, index=years)

    df_hw = pd.DataFrame({
        'words': [0 for year in years]
    }, index=years)

    # for each year determine top 10 keywords
    top_keywords_by_year_sc = {}
    top_keywords_by_year_sw = {}
    top_keywords_by_year_hw = {}
    for year in years:
        
        # initialize the top 10
        top_keywords_sc = [('', 0) for i in range(10)]
        top_keywords_sw = [('', 0) for i in range(10)]
        top_keywords_hw = [('', 0) for i in range(10)]

        # grab the histograms
        hist_sc = hist_by_year_sc[year]
        hist_sw = hist_by_year_sw[year]
        hist_hw = hist_by_year_hw[year]

        # Sort SC #
        for word in hist_sc:
            count = hist_sc[word]

            # compare the current word to the words on the list
            for i, (keyword, keycount) in enumerate(top_keywords_sc):
                
                # if the count is greater, then push all the other words back 
                # store this one
                # then break
                if count >= keycount:
                    for j in range(9, i, -1):
                        top_keywords_sc[j] = top_keywords_sc[j-1]
                    
                    top_keywords_sc[i] = (word, count)

                    break

        # Sort SW #
        for word in hist_sw:
            count = hist_sw[word]

            # compare the current word to the words on the list
            for i, (keyword, keycount) in enumerate(top_keywords_sw):
                
                # if the count is greater, then push all the other words back 
                # store this one
                # then break
                if count >= keycount:
                    for j in range(9, i, -1):
                        top_keywords_sw[j] = top_keywords_sw[j-1]
                    
                    top_keywords_sw[i] = (word, count)

                    break

        # Sort HW #
        for word in hist_hw:
            count = hist_hw[word]

            # compare the current word to the words on the list
            for i, (keyword, keycount) in enumerate(top_keywords_hw):
                
                # if the count is greater, then push all the other words back 
                # store this one
                # then break
                if count >= keycount:
                    for j in range(9, i, -1):
                        top_keywords_hw[j] = top_keywords_hw[j-1]
                    
                    top_keywords_hw[i] = (word, count)

                    break

        # store in datastructure for this year
        top_keywords_by_year_sc[year] = top_keywords_sc
        top_keywords_by_year_sw[year] = top_keywords_sw
        top_keywords_by_year_hw[year] = top_keywords_hw

        df_sc['words'][year] = ", ".join([word for (word, count) in top_keywords_sc])
        df_sw['words'][year] = ", ".join([word for (word, count) in top_keywords_sw])
        df_hw['words'][year] = ", ".join([word for (word, count) in top_keywords_hw])

    df_sc.to_csv('data/analysis/sc_term_freq.csv')
    df_sw.to_csv('data/analysis/sw_term_freq.csv')
    df_hw.to_csv('data/analysis/hw_term_freq.csv')

    # create and sort a list of keywords with freqs
    top_keywords_global_sc = [(word, hist_global_sc[word]) for word in hist_global_sc.keys()]
    top_keywords_global_sw = [(word, hist_global_sw[word]) for word in hist_global_sw.keys()]
    top_keywords_global_hw = [(word, hist_global_hw[word]) for word in hist_global_hw.keys()]

    top_keywords_global_sc.sort(key=lambda x: x[1], reverse=True)
    top_keywords_global_sw.sort(key=lambda x: x[1], reverse=True)
    top_keywords_global_hw.sort(key=lambda x: x[1], reverse=True)

    # print them out to a local csv file
    strOut_sc = ''
    for (word, count) in top_keywords_global_sc:
        strOut_sc += word + ', ' + str(count) + '\n'
    with open('data/analysis/sc_global_keyword_hist.csv', mode='w') as file:
        file.write(strOut_sc)

    strOut_sw = ''
    for (word, count) in top_keywords_global_sw:
        strOut_sw += word + ', ' + str(count) + '\n'
    with open('data/analysis/sw_global_keyword_hist.csv', mode='w') as file:
        file.write(strOut_sw)

    strOut_hw = ''
    for (word, count) in top_keywords_global_hw:
        strOut_hw += word + ', ' + str(count) + '\n'
    with open('data/analysis/hw_global_keyword_hist.csv', mode='w') as file:
        file.write(strOut_hw)

# given this list of keywords of interest
def term_freq_analysis_narrow(data_labeled_from_model_addr='data/recall_labeled_from_model.csv',
                        data_labeled_addr='data/recall_labeled.csv'):

    software_kwoi = [
        set(['anomaly']),
        set(['image', 'imaging']),
        set(['interface', 'gui']),
        set(['version', 'v']),
        set(['protocol', 'message'])
    ]
    hardware_kwoi = [
        set(['defective', 'damaged']),
        set(['battery', 'power', 'charge', 'energy', 'voltage', 'charging', 'charger']),
        set(['board', 'circuit', 'capacitor', 'wiring', 'pcb']),
        set(['alarm']),
        set(['monitor', 'display'])
    ]
    security_kwoi = [
        set(['error']),
        set(['sent', 'transfer', 'recieved']),
        set(['deleted', 'corrupted']),
        set(['anomaly']),
        set(['data', 'file', 'information', 'disk', 'archive', 'record'])
    ]

    # read both the samples labeled
    df_man = pd.read_csv(data_labeled_addr)
    df_auto = pd.read_csv(data_labeled_from_model_addr)

    # combine them
    data = pd.concat([df_man, df_auto], ignore_index=True)

    # determine number of years
    # it should be 17, from 2002 to 2018
    years = data['YEAR']
    years = years.drop_duplicates()
    numYears = years.shape[0]

    # seperate hist for each class
    # determing most popular words for each year
    # store in dict with year key
    hist_by_year_sc = {}
    hist_by_year_sw = {}
    hist_by_year_hw = {}
    for year in years:
        hist_by_year_sc[year] = [0, 0, 0, 0, 0]
        hist_by_year_sw[year] = [0, 0, 0, 0, 0]
        hist_by_year_hw[year] = [0, 0, 0, 0, 0]

    # go through each sample
    # update hist by year and class
    for i, row in data.iterrows():

        # read the row
        year = row['YEAR']
        sentance = row['MANUFACTURER_RECALL_REASON']
        label_SC = row['SC']
        label_SW = row['SW']
        label_HW = row['HW']

        wordSet = set(tokenize(sentance))

        # for each class update the hist
        if label_SC == 1:

            # Read hist into local mem
            hist = hist_by_year_sc[year]

            # For each keyword of interest if there exists some non-zero overlap in words 
            # increment the counter for that set
            for j, kwoi_j in enumerate(security_kwoi):
                if len(wordSet.intersection(kwoi_j)) > 0:
                    hist[j]+=1

            # Write back to main mem
            hist_by_year_sc[year] = hist
            
        if label_SW == 1:

            # Read hist into local mem
            hist = hist_by_year_sw[year]

            # For each keyword of interest if there exists some non-zero overlap in words 
            # increment the counter for that set
            for j, kwoi_j in enumerate(software_kwoi):
                if len(wordSet.intersection(kwoi_j)) > 0:
                    hist[j]+=1

            # Write back to main mem
            hist_by_year_sw[year] = hist
            
        if label_HW == 1:

            # Read hist into local mem
            hist = hist_by_year_hw[year]

            # For each keyword of interest if there exists some non-zero overlap in words 
            # increment the counter for that set
            for j, kwoi_j in enumerate(hardware_kwoi):
                if len(wordSet.intersection(kwoi_j)) > 0:
                    hist[j]+=1

            # Write back to main mem
            hist_by_year_hw[year] = hist
            
    # dataframes to be filled for each class and output of this function
    obj = {}
    for j, kwoi_j in enumerate(security_kwoi):
        wordStr = ",".join(kwoi_j)
        obj[wordStr] = [hist_by_year_sc[year][j] for year in years]
    df_sc = pd.DataFrame(obj, index=years)

    obj = {}
    for j, kwoi_j in enumerate(software_kwoi):
        wordStr = ",".join(kwoi_j)
        obj[wordStr] = [hist_by_year_sw[year][j] for year in years]
    df_sw = pd.DataFrame(obj, index=years)

    obj = {}
    for j, kwoi_j in enumerate(hardware_kwoi):
        wordStr = ",".join(kwoi_j)
        obj[wordStr] = [hist_by_year_hw[year][j] for year in years]
    df_hw = pd.DataFrame(obj, index=years)

    df_sc.to_csv('data/analysis/sc_termofinterest_freq.csv')
    df_sw.to_csv('data/analysis/sw_termofinterest_freq.csv')
    df_hw.to_csv('data/analysis/hw_termofinterest_freq.csv')


if __name__ == "__main__":
    
    # classify_samples()

    term_freq_analysis_narrow()

    #measure_stability(reupsample=True, ratio_noncomp_samples=0.0)
    #measure_stability(reupsample=True, ratio_noncomp_samples=1.0)

    #sweep_confidence_online_weight_adjust(0.0, 0.15, 0.05)

    #show_top_keywords()

    #sweep_performance(ratio_noncomp_samples=0.5)

    #sweep_alphas(5, 40, 5)

    # Labeler().run()

    #data_labeled = pd.read_csv('data/recall_labeled.csv')
    #data_ss_labeled = pd.read_csv('data/data-ss-labeled_800.csv')

    #print("--- True Labels ---")
    #print_stats(data_labeled)

    #print("--- Model Labels ---")
    #print_stats(data_ss_labeled)

    # test_performance(alpha=0, fold_count=8, sample_count=567, ratio_noncomp_samples=0.76, addr_out='data/performance/test_out.csv')
