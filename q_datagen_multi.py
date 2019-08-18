# -*- coding: utf-8 
## @package q_datagen_multi
# q_datagen_multi -> <MSSA of Windowed Timeseries + FeatureExtractor_symbol> -> q_pretrainer_neat -> <Pre-Trained ANN model> 
# -> q_agent(FeatureExtractor+ANN_model) -> <Performance> -> q_mlo -> <Optimal FeatureExtractor> <=> AgentAction= [<dir>,<symbol>,<TP>,<SL>]
#
#  Version "multi" controls the fx-environment, with observations containing: hlc, <MSSA of all features>.
#
#  Creates a dataset with MSSA of the timeseries loaded from a CSV file,
#  the dataset contains the transformation of each of the given rows.

#  Also exports the feature extractor (transform to be applied to data)

#  Applies and exports Box-Cox transform for gaussian aproximation and standarization into all signals.
#
#  NOTE: The tested input dataset used 5 symbols, with consecutive features for each symbol, in the following order: h,l,c,v,indicators
#
# INSTALLATION NOTE: For importing new environment in ubuntu run, export PYTHONPATH=${PYTHONPATH}:/home/[your username]/gym-forex/

import pandas as pd
import numpy as np
from numpy import genfromtxt
from numpy import concatenate
from collections import deque
import sys
import csv 
from sklearn import preprocessing
import matplotlib.pyplot as plt
from joblib import dump, load
from pymssa import MSSA
import struct
print(struct.calcsize("P") * 8)

# main function
# parameters: state/action code: 0..3 for open, 4..7 for close 
if __name__ == '__main__':
    
    # command line arguments
    # argument 1 = input dataset in csv format, contains num_obs observations(rows) of  the input features (columns)
    csv_f =  sys.argv[1]
    # argument 2 = output component dataset in csv format, contains (num_obs-window_size) rows with the first n_components per feature(columns), standarized values
    c_out_f = sys.argv[2]
    # argument 3 = output trimmed dataset in csv format, contains the hlc columns of the original input dataset without the first window of observations for 1-to 1 relation with output(for use in agent)
    t_out_f = sys.argv[3]
    # argument 4 = prefix for the standarization data files 
    s_out_f = sys.argv[4]
    # argument 5 = window_size used for calculating the components for each observation (row) of the input dataset
    p_window_size = int(sys.argv[5])
    # argument 6 = n_components the number of components exported in the output component dataset
    p_n_components = int(sys.argv[6])
    
    # inicializations
    # Number of training signals
    num_symbols = 5
    # number of features per symbol
    features_per_symbol = 29
    
    # load csv file, The file must contain 16 cols: the 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v, 5 = MoY, 6 = DoM, 7 = DoW, 8 = HoD, 9 = MoH, ..<6 indicators>
    my_data = genfromtxt(csv_f, delimiter=',')
    # get the number of observations
    num_ticks = len(my_data)
    num_columns = len(my_data[0])
    
    # standarize the data and export normalization data using StandardScaler and joblib.dump
    pre = preprocessing.StandardScaler()
    # gets the standarization statistics from 75% of input dataset
    ts_data = my_data[ 0: (num_ticks*3)//4 , :]
    pre.fit(ts_data) 
    # standarize the whole dataset
    s_data = pre.transform(my_data) 
    print("Saving pre-processing.StandardScaler() settings for the generated dataset")
    dump(pre, s_out_f+'.standardscaler')  
    # perform MSSA on standarized data
    print("Performing MSSA on filename="+ str(csv_f) + ", n_components=" + str(p_n_components) + ", window_size=" + str(p_window_size))
    mssa = MSSA(n_components=p_n_components, window_size=p_window_size)
    mssa.fit(s_data.astype(np.int64))
    # TODO: graficar componentes acumulativos desde 1 hasta n_components, comparados con el dataset estandarizado
    # for the 5th and the next components, save plots containing the original and cummulative timeseries for the first data column 
    cumulative_recon = np.zeros_like(s_data.iloc[:, 0].values)
    for comp in range(mssa.rank_):  
        fig, ax = plt.subplots(figsize=(18, 7))
        current_component = mssa.components_[0, :, comp]
        cumulative_recon = cumulative_recon + current_component
        ax.plot(s_data.index, s_data.iloc[:, 0].values, lw=3, alpha=0.2, c='k', label=s_data.columns[0])
        ax.plot(s_data.index, cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
        ax.plot(s_data.index, current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
        ax.legend()
        fig.savefig('mssa_' + str(comp) + '.png', dpi=600)
        
    # TODO: Save the datasets and the rank matrix
    print("Finished generating extended dataset.")
    print("Done.")
    
    
