# -*- coding: utf-8 
## @package q_datagen
# q_datagen -> <Windowed Datasets whith reward> -> q_pretrainer -> <Pre-Trained model> -> q_agent -> <Performance> -> q_mlo -> <Optimal Net per Action>
# usage: python3 q-datagen <csv_dataset> <output> <window_size> <min_TP> <max_TP> <min_SL> <max_SL> 

#  Creates a dataset with observations and the reward for an action (first command line parameter)
#  assumes an order can be open in each tick and calculates the StopLoss, TakeProfit and Volume parameters of an 
#  optimal order (TPr, SLr, Vol) with the minimum and maximum of these parameters (max_TP, max_SL, max_Vol) given via 
#  command-line and method parameters. Also transforming the action values so the 
#  training signals are continuous around 0 (for testing if this transform improves q-pretrainer performance). 
#  applies Box-Cox transform for gaussian aproximation and standarization into all signals.
#
#  Also calculate the same for the returned values of the features and generate 
# independent classification signals for: direction(0,1) ,  
#  TP>max_TP/2, SL>max_SL/2, volume>max_volume/2, ema adelantado (#TODO: Buscar columna de EMA en mql script)
#  y se asume que se el dataset de entrada tiene valores no-retornados, incluyendo
#  las dimensiones por candlestick para input selection.
#
# Continuous actions
#  action = 0: TP_buy   action = 3: TP_sell     
#  action = 1: SL_buy   action = 4: SL_sell
#  action = 2: dInv_buy action = 5: dInv_sell
#
# Discrete actions
#  action = 6: TP_buy   action = 9: TP_sell     
#  action = 7: SL_buy   action = 10: SL_sell
#  action = 8: dInv_buy action = 11: dInv_sell
#  action = 12: direction
#
# Differentiable continuous actions
#  action = 13: forward EMA return 
#
#  For importing new environment in ubuntu run, export PYTHONPATH=${PYTHONPATH}:/home/[your username]/gym-forex/
import numpy as np
from numpy import genfromtxt
from numpy import shape
from numpy import concatenate
from numpy import transpose
from numpy import concatenate
from collections import deque
import sys
from itertools import islice 
import csv 
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn import svm
from joblib import dump, load

def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.mutual_info_regression(X,Y,discrete_features=False) #center=True (the default) would not work ("ValueError: center=True only allowed for dense data") but should presumably work in general

# search_order function: search for the optimal search and buy order in the given time window,
def search_order(action, window, min_TP, max_TP, min_SL, max_SL, min_dInv, max_dInv):
    max = -9999999
    min = 9999999
    max_i = -1
    min_i = -1
    dd_max = 999999
    dd_min = -9999999
    dd_max_i = -1
    dd_min_i = -1
    open_index = 0
    # busca max y min
    start_tick = 0
    # direction es 1: buy, -1:sell, 0:nop
    direction=0
    # En cada tick verificar si la mejor orden es buy o sell comparando el reward (profit-dd) en buy y sell y verificando que el dd sea menor a max_SL
    open_buy = window[0][0]
    open_buy_index = 0
    open_sell = window[0][1]
    open_sell_index = 0
    # search for max/min and drawdown for open buy and sell    
    for index, obs in enumerate(window):
        if index <= max_dInv:
            # compara con el low de cada obs (worst case), index 1
            if (max < obs[1]): 
                max = obs[1]
                max_i = index
            # compara con el high de cada obs (worst case), index 0
            if min > obs[0]: 
                min = obs[0]
                min_i = index  
    # busca dd (max antes de min o vice versa)
    for index, obs in enumerate(window):
        # busca min antes de max compara con el low de cada obs (worst case), index 1
        if (dd_max > obs[1]) and (index <= max_i): 
            dd_max = obs[1]
            dd_max_i = index
        # compara con el high de cada obs (worst case), index 0
        if (dd_min < obs[0]) and (index <= min_i): 
            dd_min = obs[0]
            dd_min_i = index

    # print("s=",stateaction, "oi=",open_index, " max=",max," max_i=",max_i," dd_max=",dd_max, " dd_max_i=", dd_max_i)
    # print("s=",stateaction, "oi=",open_index, " min=",min," min_i=",min_i," dd_min=",dd_min, " dd_min_i=", dd_min_i)
    pip_cost = 0.00001

    # profit_buy = (max-open)/ pip_cost
    profit_buy  = (max-open_buy)/pip_cost
    # dd_buy = (open-min) / pip_cost
    dd_buy = (open_buy-dd_max) / pip_cost
    # reward_buy = profit - dd
    reward_buy = profit_buy / (dd_buy + 1)
    # profit_sell = (open-min)/ pip_cost
    profit_sell  = (open_sell-min)/pip_cost
    # dd_sell = (max-open) / pip_cost
    dd_sell = (dd_min-open_sell) / pip_cost
    # reward_sell = profit - dd
    reward_sell = profit_sell / (dd_sell + 1)
    return open_sell_index, open_buy_index, max, min, max_i, min_i, profit_buy, dd_buy, dd_max_i, reward_buy, profit_sell, dd_sell, dd_min_i, reward_sell 

# getReward function: calculate the reward for the selected state/action in the given time window(matrix of observations) 
# @param: stateaction = state action code (0..3) open order, (4..7) close existing
# @param: window = High, Low, Close, nextOpen timeseries
def get_reward(action, window, min_TP, max_TP, min_SL, max_SL, min_dInv, max_dInv):
    # if the draw down of the highest between buy and sell profits, is more than max_SL, 
    # search for the best order before the dd
    last_dd = max_SL
    i_dd = max_dInv
    reward_buy  = 0.0
    reward_sell = 0.0
    direction = 0
    # the first 3 actions for buy:  0:TP, 1:SL and 2:dInv
    if action < 3:
        # search for the best buy order on the current window
        while (last_dd >= max_SL) and (reward_buy <= reward_sell):
            open_sell_index, open_buy_index, max, min, max_i, min_i, profit_buy, dd_buy, dd_max_i, reward_buy, profit_sell, dd_sell, dd_min_i, reward_sell = search_order(action, window, min_TP, max_TP, min_SL, max_SL, min_dInv, i_dd)
            if reward_buy > reward_sell:
                last_dd = dd_buy 
                i_dd = dd_max_i
            else:
                last_dd = dd_sell
                i_dd = dd_min_i
            if i_dd <= min_dInv:
                break
        # Buy continuous actions (0:TP, 1:SL, 2:dInv proportional to max vol), else search the next best buy order before the dd 
        if reward_buy > reward_sell:
            direction = 1
            # case 0: TP buy, reward es el profit de buy
            # en clasification, para tp y sl ya dos niveles:0=bajo y 1=alto
            if action == 0:
                if profit_buy < min_TP:
                    reward = 0
                if profit_buy > max_TP:
                    reward = 1
                else:
                    reward = profit_buy / max_TP
            # case 1: SL buy, if dir = buy, reward es el dd de buy 
            elif action == 1:
                if dd_buy < min_SL:
                    reward = 0
                if dd_buy > max_SL:
                    reward = 1
                else:
                    reward = dd_buy / max_SL    
            # case 2: dInv, if dir = buy, reward es el index del max menos el de open.
            elif action == 2:
                reward = (max_i - open_buy_index) / max_dInv
                if  (max_i - open_buy_index) < min_dInv:
                    reward = 0
            return {'reward':reward, 'profit':profit_buy, 'dd':dd_buy ,'min':min ,'max':max, 'direction':direction}
        # sino, retorna 0 a todas las acciones
        else:
            return {'reward':0, 'profit':profit_buy, 'dd':dd_buy ,'min':min ,'max':max, 'direction':0}
    # the actions for sell:  3:TP, 4:SL and 5:dInv
    if (action >= 3) and (action <6):
        # search for the best sell order on the current window
        while (last_dd >= max_SL) and (reward_sell <= reward_buy):
            open_sell_index, open_buy_index, max, min, max_i, min_i, profit_buy, dd_buy, dd_max_i, reward_buy, profit_sell, dd_sell, dd_min_i, reward_sell = search_order(action, window, min_TP, max_TP, min_SL, max_SL, min_dInv, i_dd)
            if reward_sell> reward_buy :
                last_dd = dd_sell 
                i_dd = dd_min_i
            else:
                last_dd = dd_buy
                i_dd = dd_max_i
            if i_dd <= min_dInv:
                break
        # Sell continuous actions (3:TP, 4:SL, 5:dInv proportional to max vol), else search the next best buy order before the dd 
        if reward_sell > reward_buy:
            direction = -1
            # case 0: TP sell, reward es el profit de sell
            if action == 3:
                if profit_sell < min_TP:
                    reward = 0
                if profit_sell > max_TP:
                    reward = 1
                else:
                    reward = profit_sell / max_TP
            # case 1: SL sell, if dir = sell, reward es el dd de sell 
            elif action == 4:
                if dd_sell < min_SL:
                    reward = 0
                if dd_sell > max_SL:
                    reward = 1
                else:
                    reward = dd_sell / max_SL    
            # case 2: dInv, if dir = sell, reward es el index del max menos el de open.
            elif action == 5:
                reward = (min_i - open_sell_index) / max_dInv
                if  (min_i - open_sell_index) < min_dInv:
                    reward = 0
            return {'reward':reward, 'profit':profit_sell, 'dd':dd_sell ,'min':min ,'max':max, 'direction':direction}
        # sino, retorna 0 a todas las acciones
        else:
            return {'reward':0, 'profit':profit_sell, 'dd':dd_sell ,'min':min ,'max':max, 'direction':0}
   # Continuous indicators = 6:rEMA, 7:rRSI, 8:rnEMA,9 rnRSI
    if action == 6:
        # RETURN DE EMA no normalizado (EMAf-EMAini) ADELANTADO 4 dias (TODO: Probar con period =7 y no 14 como el actual dataset)
        return {'reward':(window[3][7] - window[2][7]), 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':0}
    if action == 7:
        # RETURN DE  RSI no normalizado (RSIf - RSI ACTUAL) ADELANTADO 1 día, strat: cierra en cambio de signo de pendiente 
        return {'reward':(window[1][8] - window[0][8]), 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':0}
    if action == 8:
        # RETURN DE EMA normalizado (EMAf-EMAini)/EMAini ADELANTADO 4 dias (TODO: Probar con period =7 y no 14 como el actual dataset)
        return {'reward':(window[3][7] - window[2][7])/window[2][7], 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':0}
    if action == 9:
        # RETURN DE  RSI normalizado (RSIf - RSI ACTUAL)/RSIActual ADELANTADO 1 día, strat: cierra en cambio de signo de pendiente 
        return {'reward':(window[1][8] - window[0][8])/window[0][8], 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':0}

    if action == 10:
        # RETURN DE MACD ADELANTADO 10 ticks (TODO: Probar otros valores para etrategia de prueba)
        if (window[11][9] - window[10][9]) > 0:
            rew = 1
        else:
            rew = 0
        return {'reward': rew, 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':rew}

    # case 11: SL buy, if dir = buy, reward es el dd de buy 
    if action == 11:
        # RETURN DE MACD ADELANTADO 11 ticks (TODO: Probar otros valores para etrategia de prueba)
        if (window[12][9] - window[11][9]) > 0:
            rew = 1
        else:
            rew = 0
        return {'reward': rew, 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':rew}

    # case 12: dInv, if dir = buy, reward es el index del max menos el de open.
    if action == 12:
        # RETURN DE MACD ADELANTADO 12 ticks (TODO: Probar otros valores para etrategia de prueba)
        if (window[13][9] - window[12][9]) > 0:
            rew = 1
        else:
            rew = 0
        return {'reward': rew, 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':rew}

    if action == 13:
        # RETURN DE MACD ADELANTADO 13 ticks (TODO: Probar otros valores para etrategia de prueba)
        if (window[14][9] - window[13][9]) > 0:
            rew = 1
        else:
            rew = 0
        return {'reward': rew, 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':rew}
    
    if action == 14:
        # RETURN DE MACD ADELANTADO 14 ticks (TODO: Probar otros valores para etrategia de prueba)
        if (window[15][9] - window[14][9]) > 0:
            rew = 1
        else:
            rew = 0
        return {'reward': rew, 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':rew}
    
    elif action == 15:
        # RETURN DE MACD ADELANTADO 15 ticks (TODO: Probar otros valores para etrategia de prueba)
        if (window[16][9] - window[15][9]) > 0:
            rew = 1
        else:
            rew = 0
        return {'reward': rew, 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':rew}
    
    if action == 16:
        # RETURN DE MACD ADELANTADO 16 ticks (TODO: Probar otros valores para etrategia de prueba)
        if (window[17][9] - window[16][9]) > 0:
            rew = 1
        else:
            rew = 0
        return {'reward': rew, 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':rew}
    if action == 17:
        # RETURN DE MACD ADELANTADO 17 ticks (TODO: Probar otros valores para etrategia de prueba)
        if (window[18][9] - window[17][9]) > 0:
            rew = 1
        else:
            rew = 0
        return {'reward': rew, 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':rew}
    if action == 18:
        # RETURN DE MACD ADELANTADO 26 ticks (TODO: Probar otros valores para etrategia de prueba)
        # tiene max balance 800-16k en 1y pero error=0.278 con indicator_period=77 sin short-long term data
        if (window[27][9] - window[26][9]) > 0:
            rew = 1
        else:
            rew = 0
        return {'reward': rew, 'profit':0, 'dd':0 ,'min':0 ,'max':0, 'direction':rew}

# main function
# parameters: state/action code: 0..3 for open, 4..7 for close 
if __name__ == '__main__':
    # initializations
    csv_f =  sys.argv[1]
    out_f = sys.argv[2]
    window_size = int(sys.argv[3])
    min_TP = int(sys.argv[4])
    max_TP = int(sys.argv[5])
    min_SL = int(sys.argv[6])
    max_SL = int(sys.argv[7])
    min_dInv = 2
    max_dInv = window_size
    
    # Number of training signals
    num_signals = 19
    
    # feature selection threshold
    selection_score = 0.2
    
    # load csv file, The file must contain 16 cols: the 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v, 5 = MoY, 6 = DoM, 7 = DoW, 8 = HoD, 9 = MoH, ..<6 indicators>
    my_data = genfromtxt(csv_f, delimiter=',')
    # returned values (vf-vi)/vi
    my_data_r = genfromtxt(csv_f, delimiter=',')
    # get the number of observations
    num_ticks = len(my_data)
    num_columns = len(my_data[0])
    
    # initialize maximum and minimum
    max = num_columns * [-999999.0]
    min = num_columns * [999999.0]
    promedio = num_columns * [0.0]
    
    # calcula el return usando el valor anterior de cada feature y max,min para headers de output (TODO: VERIFICAR INVERSA DE PowerTransformer Y GUARDAR DATOS RELEVANTES EN LUGAR DE MAX, MIN EN HEADERS DE OUTPUT)
    for i in range(1, num_ticks):        
        for j in range(0, num_columns):
            # asigna valor en matrix para valores retornados
            if my_data[i-1,j] != 0:
                my_data_r[i,j] = (my_data[i,j] - my_data[i-1,j]) 
            else:
                my_data_r[i,j] = (my_data[i,j] - my_data[i-1,j]) 
    
    # concatenate the data and the returned values
    my_data = concatenate((my_data, my_data_r), axis=1)
    
    # calcula  max,min para headers de output (TODO: VERIFICAR INVERSA DE PowerTransformer Y GUARDAR DATOS RELEVANTES EN LUGAR DE MAX, MIN EN HEADERS DE OUTPUT)
    for i in range(1, num_ticks):        
        for j in range(0, num_columns):
            # actualiza max y min
            if my_data[i, j] > max[j]:
                max[j] = my_data[i, j]
            if my_data[i, j] < min[j]:
                min[j] = my_data[i, j]
                # incrementa acumulador
                promedio[j] = promedio[j] + my_data[i, j]   
    
    # window = deque(my_data[0:window_size-1, :], window_size)
    window = deque(my_data[0:window_size-1, :], window_size)
    window_future = deque(my_data[window_size:(2*window_size)-1, :], window_size)
    
    # inicializa output   
    output = []
    print("Generating dataset with " + str(len(my_data[0, :])) + " features with " + str(window_size) + " past ticks per feature and ",num_signals," reward related features. Total: " + str((len(my_data[0, :]) * window_size)+num_signals) + " columns.  \n" )
    # initialize window and window_future para cada tick desde 0 hasta window_size-1
    for i in range(1, window_size+1):
        tick_data = my_data[i, :].copy()
        tick_data_future = my_data[i+window_size, :].copy()
        # fills the training window with past data
        window.appendleft(tick_data.copy())
        # fills the future dataset to search for optimal order
        window_future.append(tick_data_future.copy())
    
    # para cada tick desde window_size hasta num_ticks - 1
    for i in range(window_size, num_ticks-window_size):
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
        for j in range (0,num_signals):
            res.append(get_reward(j, window_future, min_TP, max_TP, min_SL, max_SL, min_dInv, max_dInv)) 

        for it,v in enumerate(tick_data):
            # expande usando los window tick anteriores (traspuesta de la columna del feature en la matriz window)
            # window_column_t = transpose(window[:, 0])
            w_count = 0
            for w in window:
                if (w_count == 0) and (it==0):
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
            
        # concatenate expanded tick data per feature with reward 
        for j in range (0,num_signals):
            tick_data_r = concatenate ((tick_data_r, [res[j]['reward']])) 
        output.append(tick_data_r)
        # print('len(tick_data) = ', len(tick_data), ' len(tick_data_c) = ', len(tick_data_c))
        
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
    headers = concatenate((headers,["TPbuy_"+str(min_TP)+"_"+str(max_TP)]))        
    headers = concatenate((headers,["SLbuy_"+str(min_SL)+"_"+str(max_SL)]))        
    headers = concatenate((headers,["dInvbuy_"+str(min_dInv)+"_"+str(max_dInv)]))         
    headers = concatenate((headers,["TPsell_"+str(min_TP)+"_"+str(max_TP)]))        
    headers = concatenate((headers,["SLsell_"+str(min_SL)+"_"+str(max_SL)]))        
    headers = concatenate((headers,["dInvsell_"+str(min_dInv)+"_"+str(max_dInv)]))         
    headers = concatenate((headers,["rEMA4d"]))        
    headers = concatenate((headers,["rRSI1d"]))        
    headers = concatenate((headers,["rnEMA4d"]))        
    headers = concatenate((headers,["rnRSI1d"]))        
    headers = concatenate((headers,["cTPbuy_"+str(min_TP)+"_"+str(max_TP)]))        
    headers = concatenate((headers,["cSLbuy_"+str(min_SL)+"_"+str(max_SL)]))        
    headers = concatenate((headers,["cdInvbuy_"+str(min_dInv)+"_"+str(max_dInv)]))         
    headers = concatenate((headers,["cTPsell_"+str(min_TP)+"_"+str(max_TP)]))        
    headers = concatenate((headers,["cSLsell_"+str(min_SL)+"_"+str(max_SL)]))        
    headers = concatenate((headers,["cdInvsell_"+str(min_dInv)+"_"+str(max_dInv)]))  
    headers = concatenate((headers,["cEMA10d"]))  
    headers = concatenate((headers,["cRSI1d"]))  
    headers = concatenate((headers,["cEMAdiff10d"]))  
    # Applies YeoJohnson transform with standarization (zero mean/unit variance normalization) to each column of output (including actions?)
    pt = preprocessing.PowerTransformer()
    to_t = np.array(output)
    to_tn = to_t[: , 0: (2 * num_columns * window_size)+10]
    # probando con min-max antes de prowertransform en las 3 últimas features
    #from sklearn.preprocessing import MinMaxScaler
    #sc = MinMaxScaler(feature_range = (0, 1))
    #training_set_scaled = sc.fit_transform(to_tn)
    
    output_bt = pt.fit_transform(to_tn)
    # save the preprocessing settings
    print("saving pre-processing.PowerTransformer() settings for the generated dataset")
    dump(pt, out_f+'.powertransformer')

    output_bc = concatenate((output_bt,to_t[: , ((2 * num_columns * window_size) + 10) : ((2 * num_columns * window_size) + num_signals)]),1)
    # plots  the data selection graphic
    fig=plt.figure(1)
    plt.clf()

    X = output_bt[:,0:2*num_columns*window_size]
    # hace la selección de características respecto a predicción del RSI (17)
    y = to_t[: , ((2 * num_columns * window_size) + 16) ]
    X_indices = np.arange(X.shape[-1])

    # #############################################################################
    # Univariate feature selection with mutual information (best than ANOVA for non-linear inputs) for feature scoring
    # We use the default selection function: the 10=0.55,20=0.54, 25=0.51 30=0.52, 50=0.53  most significant features
    selector = SelectPercentile(mutual_info_classif, percentile=25)
    selector.fit(X, y)
    scores = selector.scores_
    scores /= scores.max()
    plt.bar(X_indices, scores, width=0.7,
            label=r'Mutual information ', color='c',
            edgecolor='black')
    plt.title("Feature Selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    fig.savefig('mutual_information.png')
    plt.show()
    #scaler = preprocessing.StandardScaler()
    #output_bc = scaler.fit_transform(output_b)
    mask = concatenate((selector.get_support().copy(), np.full(num_signals, True) ))
    # sin feature selection, e=0.37
    # con feature selection, e= TODO 
    # removes all features with less than selection_score
    accum_r = 0
    # busca hasta 2*num_columns porque también busca en los returns
    for i in range(0,2*num_columns):
        if scores[i*window_size] < selection_score:
            print("removed feature: ",i)
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
    dump(mask, out_f+'.feature_selection_mask')
    
    # Save output_bc to a file
    with open(out_f , 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(headers_b)
        wr.writerows(output_b)
    print("Finished generating extended dataset.")
    print("Done.")
    
    
