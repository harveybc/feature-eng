# -*- coding: utf-8 
## @package q_datagen
# q_datagen -> <Windowed Datasets whith reward> -> q_pretrainer -> <Pre-Trained model> -> q_agent -> <Performance> -> q_mlo -> <Optimal Net per Action>
# usage: python3 q-dataset <stateaction> <dataset> <output>
#
#  Creates a dataset with observations and the reward for an action (first command line parameter)
#  assumes an order can be open in each tick and calculates the StopLoss, TakeProfit and Volume parameters of an 
#  optimal order (TPr, SLr, Vol) with the maximum of these parameters (max_TP, max_SL, max_Vol) given via 
#  command-line and method parameters. 
#
#  action = 0: TP --- TPr = SL/max_TP 
#  action = 1: SL --- SLr = SL/max_SL
#  action = 2: dInv/max_dInv --- Vol = max_Vol / dInv con abs(dInv >= 1)
#
#  For importing new environment in ubuntu run, export PYTHONPATH=${PYTHONPATH}:/home/[your username]/gym-forex/
from numpy import genfromtxt
from numpy import shape
from numpy import concatenate
from numpy import transpose
from collections import deque
import sys
from itertools import islice 
import csv 

# getReward function: calculate the reward for the selected state/action in the given time window(matrix of observations) 
# @param: stateaction = state action code (0..3) open order, (4..7) close existing
# @param: window = High, Low, Close, nextOpen timeseries
def get_reward(action, window, min_TP, max_TP, min_SL, max_SL, min_dInv, max_dInv):
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
        # compara con el low de cada obs (worst case), index 1
        if max < obs[1]: 
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
    reward_buy = profit_buy - dd_buy
    # profit_sell = (open-min)/ pip_cost
    profit_sell  = (open_sell-min)/pip_cost
    # dd_sell = (max-open) / pip_cost
    dd_sell = (dd_min-open_sell) / pip_cost
    # reward_sell = profit - dd
    reward_sell = profit_sell - dd_sell
    # calculate the direction of the optimal order to be opened in the current tick
    if reward_buy > reward_sell:
        direction = 1
        # case 0: TP, if dir = buy, reward es el profit de buy
        if action == 0:
            reward = direction * profit_buy / max_TP
            if profit_buy < min_TP:
                reward = 0
        # case 1: SL, if dir = buy, reward es el dd de buy
        elif action == 1:
            reward = direction * dd_buy / max_SL
            if dd_buy < min_SL:
                reward = 0
        # case 2: dInv, if dir = buy, reward es el index del max menos el de open.
        else:
            reward = direction * (max_i - open_buy_index) / max_dInv
            if  (max_i - open_buy_index) < min_dInv:
                reward = 0
        return {'reward':reward , 'profit':profit_buy, 'dd':dd_buy ,'min':min ,'max':max, 'direction':direction}
    elif reward_buy < reward_sell:
        direction = -1
        # case 0: TP, if dir = buy, reward es el profit de buy
        if action == 0:
            reward = direction * profit_sell / max_TP
            if profit_sell < min_TP:
                reward = 0
        # case 1: SL, if dir = buy, reward es el dd de buy
        elif action == 1:
            reward = direction * dd_sell / max_SL
            if dd_sell < min_SL:
                reward = 0
        # case 2: dInv, if dir = buy, reward es el index del max menos el de open.
        else:
            reward = direction * (min_i - open_sell_index) / max_dInv
            if (min_i - open_sell_index) < midInv:
                reward = 0
        return {'reward':reward , 'profit':profit_sell, 'dd':dd_sell ,'min':min ,'max':max, 'direction':direction}
    else:
        direction = 0
        return {'reward':0 , 'profit':0, 'dd':0 ,'min':min ,'max':max, 'direction':direction}  

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
    min_dInv = int(sys.argv[8])
    max_dInv = int(sys.argv[9])
    
    # load csv file, The file must contain 16 cols: the 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v, 5 = MoY, 6 = DoM, 7 = DoW, 8 = HoD, 9 = MoH, ..<6 indicators>
    my_data = genfromtxt(csv_f, delimiter=',')
    my_data_n = genfromtxt(csv_f, delimiter=',')
    # get the number of observations
    num_ticks = len(my_data)
    num_columns = len(my_data[0])
    
    # initialize maximum and minimum
    max = num_columns * [-999999.0]
    min = num_columns * [999999.0]
    promedio = num_columns * [0.0]
    
    # calcula max y min para normalización
    for i in range(0, num_ticks):
        # para cada columna
        for j in range(0, num_columns):
            # actualiza max y min
            if my_data[i, j] > max[j]:
                max[j] = my_data[i, j]
            if my_data[i, j] < min[j]:
                min[j] = my_data[i, j]
                # incrementa acumulador
                promedio[j] = promedio[j] + my_data[i, j]
    
    # normalize data
    for i in range(0, num_ticks):
        # para cada columna
        for j in range(0, num_columns):
            # normalize each element
            my_data_n[i, j] = (2.0 * (my_data[i, j] - min[j]) / (max[j] - min[j])) - 1.0
    
    # lee window inicial
    
    # window = deque(my_data_n[0:window_size-1, :], window_size)
    window = deque(my_data_n[0:window_size-1, :], window_size)

    # inicializa output   
    output = []
    print("Generating dataset with " + str(len(my_data_n[0, :])) + " features with " + str(window_size) + " past ticks per feature and 7 reward related features. Total: " + str((len(my_data_n[0, :]) * window_size)+7) + " columns.  \n" )
    
    # para cada tick desde window_size hasta num_ticks - 1
    for i in range(window_size, num_ticks):
        # tick_data = my_data_n[i, :].copy()
        tick_data = my_data_n[i, :].copy()
        window.append(tick_data)
    
        # calcula reward para el estado/acción especificado como primer cmdline param
        #res = getReward(int(sys.argv[1]), window, nop_delay)
        res_0 = get_reward(0, window, min_TP, max_TP, min_SL, max_SL, min_dInv, max_dInv)
        res_1 = get_reward(1, window, min_TP, max_TP, min_SL, max_SL, min_dInv, max_dInv)
        res_2 = get_reward(2, window, min_TP, max_TP, min_SL, max_SL, min_dInv, max_dInv)
        
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
            
        # concatenate expanded tick data per feature with reward and oher trading info         
        # output_row = concatenate ((tick_data_r, [res['reward']], [res['profit']], [res['dd']], [res['min']], [res['max']], [res['dd_min']], [res['dd_max']]))
        output_row = concatenate ((tick_data_r, [res_0['reward']/100000], [res_1['reward']/100000], [res_2['reward']/100000]))
        output.append(output_row)
        # print('len(tick_data) = ', len(tick_data), ' len(tick_data_c) = ', len(tick_data_c))
        
        # TODO: ADICIONAR HEADER DE CSV CON NOMBRES DE CADA COLUMNA
        if i % 100 == 0.0:
            progress = i*100/num_ticks
            sys.stdout.write("Tick: %d/%d Progress: %d%%   \r" % (i, num_ticks, progress) )
            sys.stdout.flush()
        
    #TODO: ADICIONAR MIN, MAX Y DD A OUTPUT PARA GRAFICARLOS

        
    # calculate header names as F0-0-min-max
    headers = []
    for i in range(0, num_columns):
        for j in range(0, window_size):
            headers = concatenate((headers,["F_"+str(i)+"_"+str(j)+"_"+str(min[i])+"_"+str(max[i])]))
    headers = concatenate((headers,["TP/"+str(max_TP)]))        
    headers = concatenate((headers,["SL/"+str(max_SL)]))        
    headers = concatenate((headers,["dInv/"+str(max_dInv)]))         
        
    with open(out_f , 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        # TODO: hacer vector de headers.
        wr.writerow(headers)
        wr.writerows(output)
    print("Finished generating extended dataset.")
    print("Done.")
    
    
