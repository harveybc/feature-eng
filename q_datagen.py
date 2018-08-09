# -*- coding: utf-8 
## @package q_datagen
#  
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
from collections import deque
import sys

# getReward function: calculate the reward for the selected state/action in the given time window(matrix of observations) 
# @param: stateaction = state action code (0..3) open order, (4..7) close existing
# @param: window = High, Low, Close, nextOpen timeseries
def getReward(stateaction, window):
    w_shape = window.shape
    print ("Shape: rows=",w_shape[0]," cols=",w_shape[1])
    # busca max y su dd (min antes de max)
    
    # busca min y su dd (max antes de min)
    
    pip_cost = 0.00001
    # case 0: Open Buy, previous state = no order opened 
    # (reward=ganancia-dd en pips si se abre ahora y se cierra en el mejor caso
    if stateaction == 0:
        # toma como open el high para buy (peor caso)
        open_price = window[0, 0]
        # buscar el máximo y su index
        # buscar el mínimo antes del máximo
        # profit = (max-open)/ pip_cost
        # dd = (open-min) / pip_cost
        # profit_pips = ((Low - open_price) / pip_cost)
        # reward = profit - dd

# main function
# parameters: state/action code: 0..3 for open, 4..7 for close 
if __name__ == '__main__':
    # initializations
    window_size = 300
    csv_f =  sys.argv[2]
    #out_f = sys.argv[3]
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
    for i in range(0, num_ticks - 1):
        # para cada columna
        for j in range(0, num_columns - 1):
            # actualiza max y min
            if my_data[i, j] > max[j]:
                max[j] = my_data[i, j]
            if my_data[i, j] < min[j]:
                min[j] = my_data[i, j]
                # incrementa acumulador
                promedio[j] = promedio[j] + my_data[i, j]
    
    # normalize data
    for i in range(0, num_ticks - 1):
        # para cada columna
        for j in range(0, num_columns - 1):
            my_data_n[i, j] = (2.0 * (my_data[i, j] - min[j]) / (max[j] - min[j])) - 1.0
    
    # lee window inicial
    window = deque(window_size * [0.0], window_size)
    for i in range(0, window_size - 1):
        window.append(my_data_n[i, :].copy())
      
    # inicializa output   
    output = []
    
    # para cada tick desde window_size hasta num_ticks - 1
    for i in range(window_size, num_ticks - 1):
        tick_data = my_data_n[i, :].copy()
        window.append(tick_data)
        
        # calcula reward para el estado/acción especificado como primer cmdline param
        reward = getReward(int(sys.argv[1]), window[0:3])
        
        # append obs, reward a output
        tick_data.append(reward)
        output.append(tick_data)