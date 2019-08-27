# -*- coding: utf-8 
## @package q_datagen_multi
# q_datagen_multi -> <MSSA of Windowed Timeseries + FeatureExtractor_symbol> -> q_pretrainer_neat -> <Pre-Trained ANN model> 
# -> q_agent(FeatureExtractor+ANN_model) -> <Performance> -> q_mlo -> <Optimal FeatureExtractor> <=> AgentAction= [<dir>,<symbol>,<TP>,<SL>]
#
#  Version "multi" controls the fx-environment, with observations containing: hlc, <MSSA of all features>.
#  Version 2 generate MSSA for 2*window_size past ticks
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
import seaborn as sns
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
    num_columns =  len(my_data[0])
    
    # standarize the data and export normalization data using StandardScaler and joblib.dump
    pre = preprocessing.StandardScaler()
    # gets the standarization statistics from 75% of input dataset
    ts_data = my_data[ 0: (num_ticks*3)//4 , :]
    pre.fit(ts_data) 
    
    # standarize the whole dataset
    s_data = pre.transform(my_data) 
    print("Saving pre-processing.StandardScaler() settings for the generated dataset")
    dump(pre, s_out_f+'.standardscaler')  
    output  = np.array([])
    # TODO: Realizar un MSSA por tick con sub_data = data[i:i+2*window_size,:] 
    # perform MSSA on standarized data
    print("Performing MSSA on filename="+ str(csv_f) + ", n_components=" + str(p_n_components) + ", window_size=" + str(p_window_size))
    segments = (num_ticks//(2*p_window_size))
    for i in range(0, 1):
        # verify if i+(2*p_window_size) is the last observation
        first = i * (2 * p_window_size)
        if (i != segments-1):
            last = (i+1) * (2 * p_window_size)
        else:
            last = num_ticks-1
        # slice the data in 2*p_window_size ticks segments
        s_data_w = s_data[first : last,:]       
        # only the first time, run svht, in following iterations, use the same n_components, without executing the svht algo
        if i==0: 
            mssa = MSSA(n_components='svht', window_size=p_window_size, verbose=True)
            mssa.fit(s_data_w)
            print("Selected Rank = ",str(mssa.rank_))
            rank = int(mssa.rank_)
        else:
            mssa = MSSA(n_components=rank, window_size=p_window_size, verbose=True)
            mssa.fit(s_data_w)
        
        np.append(output, mssa.components_)
        # show progress
        progress = i*100/segments
        print("Segment: ",i,"/",segments, "     Progress: ", progress," %" )
    # TODO: Guardar último tick de componente (actual=probar) en output_buffer
    #print("Saving matrix of size = (", str(len(output)),", ",str(len(output[0])), ", ", str(len(output[0][0])), ")")
    # save the components,
    #TODO:ERROR:  ValueError: could not broadcast input array from shape (145,240,13) into shape (145)
    
    np.save(c_out_f, output)
    # TODO: Graficar matriz de correlaciones del primero y  agrupar aditivamente los mas correlated.
    total_comps = mssa.components_[0, :, :]
    print(total_comps.shape)
    total_wcorr = mssa.w_correlation(total_comps)
    total_wcorr_abs = np.abs(total_wcorr)
    fig, ax = plt.subplots(figsize=(12,9))
    sns.heatmap(np.abs(total_wcorr_abs), cmap='coolwarm', ax=ax)
    ax.set_title('Component w-correlations')
    ax.legend()
    plt.show()
    # TODO: Estandarizar output, guardar archivo de estandarización.
        
    # TODO: Optional:  Guardar prediction de próximos n_pred ticks por component guardados como nuevas columnas de output_buffer
    
    

       
    # TODO: Save the datasets and the rank matrix
    
    print("Finished generating extended dataset.")
    print("Done.")
     
    
