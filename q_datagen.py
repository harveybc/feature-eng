# -*- coding: utf-8 
## @package q_datagen
# q_datagen -> <Windowed Datasets whith reward> -> q_pretrainer -> <Pre-Trained model> -> q_agent -> <Performance> -> q_mlo -> <Optimal Net per Action>
# usage: python3 q-dataset <stateaction> <dataset> <output>
#
#  Crates a dataset with observations and the reward for a state action (first command line parameter)
#  stateaction = 0: NoOrder/OpenBuy
#  stateaction = 1: NoOrder/OpenSell
#  stateaction = 2: NoOrder/NopBuy
#  stateaction = 3: NoOrder/NopSell
#  stateaction = 4: BuyOrder/Close
#  stateaction = 5: SellOrder/Close
#  stateaction = 6: BuyOrder/NoClose
#  stateaction = 7: SellOrder/NoClose
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
def getReward(stateaction, window, nop_delay):
    l_w = len(window)
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
    if stateaction == 0:
        # toma como open para Buy el high del tick actual para buy (peor caso)
        open = window[0][0]
        open_index = 0
    elif stateaction == 1:
        # toma como open para Sell el low del tick actual para sell (peor caso)
        open = window[0][1]
        open_index = 0
    elif stateaction == 2:
        start_tick = nop_delay
        # toma como open para nopBuy el high del mínimo antes de windowsize/2 y después de start_tick  
        for index, obs in enumerate(window):
            # busca min entre nop_delay  y len_window/2
            if (index < int(l_w/2)) and (index >= nop_delay): 
                # compara con el high de cada obs (worst case), index 0
                if min > obs[0]: 
                    min = obs[0]
                    open_index = index
        open = min
    else:
        start_tick = nop_delay
        # toma como open para nopSell el low del máximo antes de windowsize/2 y después de start_tick  
        for index, obs in enumerate(window):
            # busca min entre nop_delay  y len_window/2
            if (index < int(l_w/2)) and (index >= nop_delay): 
                # compara con el low de cada obs (worst case), index 1
                if max < obs[1]: 
                    max = obs[1]
                    open_index = index
        open = max
        
    # search for max/min and drawdown for open buy and sell    
    if stateaction < 2:
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
                
    # search for max/min and drawdown for nop open buy and sell    
    else:
        # busca min y max después de open
        for index, obs in enumerate(window):
            if index >= open_index:
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
            if index >= open_index:
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

    # case 0: Open Buy/CloseSell/nopCloseBuy, previous state = no order opened (reward=ganancia-dd) en pips si se abre ahora y se cierra en el mejor caso
    if stateaction == 0:
        # profit = (max-open)/ pip_cost
        profit  = (max-open)/pip_cost
        # dd = (open-min) / pip_cost
        dd = (open-dd_max) / pip_cost
        # reward = profit - dd
        reward = profit - dd

    # case 1: Open Sell/CloseBuy/nopCloseSell, previous state = no order opened (reward=ganancia-dd) en pips si se abre ahora y se cierra en el mejor caso
    if stateaction == 1:

        # profit = (open-min)/ pip_cost
        profit  = (open-min)/pip_cost
        # dd = (max-open) / pip_cost
        dd = (dd_min-open) / pip_cost
        # reward = profit - dd
        reward = profit - dd

    # case 2: No Open Buy, previous state = buy (reward=ganancia-dd) en pips si se abre luego de window_skip ticks y se cierra en el mejor caso
    if stateaction == 2:
        # toma como open el high para buy (peor caso)
        open = window[start_tick][0]
        # profit = (max-open)/ pip_cost
        profit  = (max-open)/pip_cost
        # dd = (open-min) / pip_cost
        dd = (open-dd_max) / pip_cost
        # reward = profit - dd
        reward = profit - dd

    # case 3: No Open Sell, previous state = state = sell (reward=ganancia-dd) en pips si se abre luego de window_skip ticks y se cierra en el mejor caso
    if stateaction == 3:
        # toma como open el high para buy (peor caso)
        open = window[start_tick][1]
        # profit = (open-min)/ pip_cost
        profit  = (open-min)/pip_cost
        # dd = (max-open) / pip_cost
        dd = (dd_min-open) / pip_cost
        # reward = profit - dd
        reward = profit - dd
        
    return {'reward':reward , 'profit':profit, 'dd':dd ,'min':min ,'max':max, 'dd_min':dd_min,'dd_max':dd_max}


# main function
# parameters: state/action code: 0..3 for open, 4..7 for close 
if __name__ == '__main__':
    # initializations
    window_size = 30
    # delay for open in nopbuy and nopsell actions
    nop_delay = 5
    csv_f =  sys.argv[2]
    out_f = sys.argv[3]
    # take_profit = int(sys.argv[4])
    
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
        res_0 = getReward(0, window, nop_delay)
        res_1 = getReward(1, window, nop_delay)
        res_2 = getReward(2, window, nop_delay)
        res_3 = getReward(3, window, nop_delay)
        
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
        output_row = concatenate ((tick_data_r, [res_0['reward']/100000], [res_1['reward']/100000], [res_2['reward']/100000], [res_3['reward']/100000]))
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
    headers = concatenate((headers,["Reward_OpenBuy/CloseSell/nopCloseBuy/100000"]))        
    headers = concatenate((headers,["Reward_OpenSell/CloseBuy/nopCloseSell/100000"]))        
    headers = concatenate((headers,["Reward_NoOpenBuy/100000"]))        
    headers = concatenate((headers,["Reward_NoOpenSell/100000"]))        
        
        
    with open(out_f , 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        # TODO: hacer vector de headers.
        wr.writerow(headers)
        wr.writerows(output)
    print("Finished generating extended dataset.")
    print("Done.")
    
    
