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

# main function
# parameters: state/action code: 0..3 for open, 4..7 for close 
if __name__ == '__main__':
    
    # command line arguments
    
    # argument 1 = input dataset in csv format, contains num_obs observations(rows) of  the input features (columns)
    csv_f =  sys.argv[1]
    # argument 2 = output component dataset in csv format, contains (num_obs-window_size) rows with the first n_components per feature(columns)
    out_f = sys.argv[2]
    # argument 3 = output trimmed dataset in csv format, contains the hlc columns of the original input dataset without the first window of observations for 1-to 1 relation with output(for use in agent)
    out_f = sys.argv[3]
    # argument 4 = window_size used for calculating the components for each observation (row) of the input dataset
    window_size = int(sys.argv[4])
    # argument 5 = n_components the number of components exported in the output component dataset
    n_components = int(sys.argv[5])
    
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
    # window = deque(my_data[0:window_size-1, :], window_size)
    window = deque(my_data[0:window_size-1, :], window_size)
    window_future = deque(my_data[window_size:(2*window_size)-1, :], window_size)
    # TODO: probar y comparar con datos de agent
    # inicializa output   
    output = []
    print("Generating dataset with " + str(len(my_data[0, :])) + " features with " + str(window_size) + " past ticks per feature and ",num_signals," reward related features. Total: " + str((len(my_data[0, :]) * window_size)+num_signals) + " columns.  \n" )
    # initialize window and window_future para cada tick desde 0 hasta window_size-1
    for i in range(0, window_size):
        tick_data = my_data[i, :].copy()
        tick_data_future = my_data[i+window_size, :].copy()
        # fills the training window with past data
        window.appendleft(tick_data.copy())
        # fills the future dataset to search for optimal order
        window_future.append(tick_data_future.copy())
    # para cada tick desde window_size hasta num_ticks - 1
    for i in range(window_size, num_ticks-window_size-1):
        # tick_data = my_data[i, :].copy()
        tick_data = my_data[i, :].copy()
        tick_data_future = my_data[i+window_size, :].copy()
        # fills the training window with past data
        window.appendleft(tick_data.copy())
        # fills the future dataset to search for optimal order
        window_future.append(tick_data_future.copy())
        # calcula reward para el estado/accion
        #res = getReward(int(sys.argv[1]), window, nop_delay)
        res = []
        # For each symbol, calculate the 5 regression signals
        for symbol in range(0, num_symbols):
            # until the 3rd prediction groups the otuputs for regression
            for j in range (0,(num_signals//num_symbols)-2):
                res.append(get_reward(num_symbols, num_signals, features_per_symbol, features_global, symbol, j, window_future, min_TP, max_TP, min_SL, max_SL, min_dInv, max_dInv));
        # For each symbol, calculate the 2 classification signals
        for symbol in range(0, num_symbols):
            # until the 3rd prediction groups the otuputs for regression
            for j in range ((num_signals//num_symbols)-2,(num_signals//num_symbols)):
                res.append(get_reward(num_symbols, num_signals, features_per_symbol, features_global, symbol, j, window_future, min_TP, max_TP, min_SL, max_SL, min_dInv, max_dInv));
            
        for it,v in enumerate(tick_data):
            # expande usando los window tick anteriores (traspuesta de la columna del feature en la matriz window)
            # window_column_t = transpose(window[:, 0])
            w_count = 0
            for w in window:
                if (w_count == 0) and (it==0):  
                #ERROR
                    window_column_t = [w[it]]
                else:
                    window_column_t = concatenate((window_column_t, [w[it]]))
                w_count = w_count + 1
            # concatenate all window_column_t for  each feature
            #if it==0:
            #    tick_data_r = window_column_t.copy()
            #else:
            #    tick_data_r = concatenate ((tick_data_r, window_column_t))
            #
            tick_data_r = window_column_t.copy()
        
        #print('i = ', i)
                # concatenate expanded tick data per feature with reward 
        for j in range (0,num_signals):
            tick_data_r = concatenate ((tick_data_r, [res[j]['reward']])) 
            
            
            
        output.append(tick_data_r)
         
        # TODO: ADICIONAR HEADER DE CSV CON NOMBRES DE CADA COLUMNA
        if i % 100 == 0.0:
            progress = i*100/num_ticks
            sys.stdout.write("Tick: %d/%d Progress: %d%%   \r" % (i, num_ticks, progress) )
            sys.stdout.flush()
        
    # calculate header names as F0-0-min-max
    headers = []
    for i in range(0, num_columns):
        for j in range(0, window_size):
            headers = concatenate((headers,["F_"+str(i)+"_"+str(j)+"_"+str(min[i])+"_"+str(max[i])]))
    for i in range(0, num_columns):
        for j in range(0, window_size):
            headers = concatenate((headers,["Fr_"+str(i)+"_"+str(j)+"_"+str(min[i])+"_"+str(max[i])]))
    for i in range(0, num_symbols):
        headers = concatenate((headers,["S"+str(i)+"TPbuy_"+str(min_TP)+"_"+str(max_TP)]))        
        headers = concatenate((headers,["S"+str(i)+"SLbuy_"+str(min_SL)+"_"+str(max_SL)]))
        headers = concatenate((headers,["S"+str(i)+"TPsell_"+str(min_TP)+"_"+str(max_TP)]))        
        headers = concatenate((headers,["S"+str(i)+"SLsell_"+str(min_SL)+"_"+str(max_SL)]))  
        headers = concatenate((headers,["S"+str(i)+"EMA10(-5)20"]))        
    for i in range(0, num_symbols):
        headers = concatenate((headers,["S"+str(i)+"cEMA10(-5)20_buy"]))        
        headers = concatenate((headers,["S"+str(i)+"cEMA10(-5)20_sell"]))        
        
    # Applies YeoJohnson transform with standarization (zero mean/unit variance normalization) to each column of output (including actions?)
    pt = preprocessing.PowerTransformer()
    #pt = preprocessing.StandardScaler()
    
    to_t = np.array(output)
    to_tn = to_t[: , 0: (2 * num_columns * window_size)]
    # probando con min-max antes de prowertransform en las 3 últimas features
    #from sklearn.preprocessing import MinMaxScaler
    #sc = MinMaxScaler(feature_range = (0, 1))
    #training_set_scaled = sc.fit_transform(to_tn)
    
    output_bt = pt.fit_transform(to_tn) 
    #output_bt = to_tn
    # save the preprocessing settings
    print("saving pre-processing.PowerTransformer() settings for the generated dataset")
    
    dump(pt, out_f+'.powertransformer')  

    output_bc = concatenate((output_bt,to_t[: , (2 * num_columns * window_size) : ((2 * num_columns * window_size) + num_signals)]),1)
    # plots  the data selection graphic
    fig=plt.figure(1)
    plt.clf()

    X = output_bt[:,0:2*num_columns*window_size]
    # hace la selección de características respecto a predicción del MACD(50) adelantado 10 ticks
    y = to_t[: , ((2 * num_columns * window_size) + 3) ]
    X_indices = np.arange(X.shape[-1])

    # #############################################################################
    # Univariate feature selection with mutual information (best than ANOVA for non-linear inputs) for feature scoring
    # We use the default selection function: the 10=0.55,20=0.54, 25=0.51 30=0.52, 50=0.53  most significant features
    selector = SelectPercentile(mutual_info_classif, percentile=25)
    selector.fit(X, y)
    scores = selector.scores_
    scores /= scores.max()
    #plt.bar(X_indices, scores, width=0.7,
    #        label=r'Mutual information ', color='c',
    #        edgecolor='black')
    #plt.title("Feature Selection")
    #plt.xlabel('Feature number')
    #plt.yticks(())
    #plt.axis('tight')
    #plt.legend(loc='upper right')
    #fig.savefig('mutual_information.png')
    #plt.show()
    mask = concatenate((selector.get_support().copy(), np.full(num_signals, True) ))
    # sin feature selection, e=0.37
    # con feature selection, e= TODO 
    # removes all features with less than selection_score
    accum_r = 0
    # busca hasta 2*num_columns porque también busca en los returns
    for i in range(0,2*num_columns):
        if (scores[i*window_size] < selection_score):
            # print("removed feature: ",i)
            accum_r+=1
            for j in range (0, window_size):
                mask[(i*window_size)+j] = False
        else:
            for j in range (0, window_size):
                mask[(i*window_size)+j] = True
    print("Total: ",accum_r," features removed (", accum_r*window_size," output columns)")
    headers_b = headers[mask]  
    output_b = output_bc[:, mask]
    # save the feature selection mask settings
    print("saving feature_selection.SelectPercentile() feature selection mask for selection_score > ",selection_score)
    dump(mask[0 : 2 * num_columns * window_size], out_f+'.feature_selection_mask')
    # Save output_bc to a file
    with open(out_f , 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(headers_b)
        wr.writerows(output_b)
    print("Finished generating extended dataset.")
    print("Done.")
    
    
