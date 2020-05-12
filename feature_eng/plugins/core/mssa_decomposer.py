# -*- coding: utf-8 -*-
"""
This File contains the MSSADecomposer class plugin. 
"""

from feature_eng.plugin_base import PluginBase
import numpy as np
from sys import exit
from pymssa import MSSA

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class MSSADecomposer(PluginBase): 
    """ Core plugin for the FeatureEng class, after initialization, saves the data and after calling the store_data method """

    def __init__(self, conf):
        """ Initializes PluginBase. Do NOT delete the following line whether you have initialization code or not. """
        super().__init__(conf)
        # Insert your plugin initialization code here.
        pass

    def parse_cmd(self, parser):
        """ Adds command-line arguments to be parsed, overrides base class """
        parser.add_argument("--forward_ticks", help="Number of forwrard ticks in the future for ema_fast", default=10, type=int)
        parser.add_argument("--ema_fast", help="Column index for ema fast", default=0, type=int)
        parser.add_argument("--ema_slow", help="Column index for ema slow", default=1, type=int)
        return parser

    def core(self, input_ds):
        """ Performs mssa_decomposition.
        """
        # get the size of the input dataset
        self.rows_d, self.cols_d = input_ds.shape
        # create an empty array with the estimated output shape
        self.output_ds = np.empty(shape=(self.rows_d-self.conf.forward_ticks, 1))
        # calculate the output
        for i in range(self.rows_d - self.conf.forward_ticks): 
            self.output_ds[i] = input_ds[i+self.conf.forward_ticks, self.conf.ema_fast]-input_ds[i, self.conf.ema_slow]
        return self.output_ds



        # performs MSSA on data
        print("Performing MSSA on filename="+ str(csv_f) + ", n_components=" + str(p_n_components) + ", window_size=" + str(p_window_size))
        segments = (num_ticks//(2*p_window_size))
        
        for i in range(0, segments):
            # verify if i+(2*p_window_size) is the last observation
            first = i * (2 * p_window_size)
            if (i != segments-1):
                last = (i+1) * (2 * p_window_size)
            else:
                last = num_ticks
            # slice the data in 2*p_window_size ticks segments
            s_data_w = s_data[first : last,:]       
            # only the first time, run svht, in following iterations, use the same n_components, without executing the svht algo
            
            if i == 0: 
                mssa = MSSA(n_components='svht', window_size=p_window_size, verbose=True)
                mssa.fit(s_data_w)
                print("Selected Rank = ",str(mssa.rank_))
                #rank = int(mssa.rank_)
                rank = int(p_n_components)
            else:
                mssa = MSSA(n_components=rank, window_size=p_window_size, verbose=True)
                mssa.fit(s_data_w)
            # concatenate otput array with the new components
            if i == 0:
                output = copy.deepcopy(mssa.components_)
            else:
                np.concatenate((output, mssa.components_), axis = 1)
                
            #TODO: concatenate grouped output 
            print("Grouping correlated components (manually set list)") 
            # use the same groups for all the features
            ts0_groups = [[0],[1],[2],[3],[4,5],[6],[7],[8],[9,10],[11],[12]]
            for j in range(0, num_columns):
                # draw correlation matrix for the first segment
                mssa.set_ts_component_groups(j, ts0_groups)
                ts0_grouped = mssa.grouped_components_[j]
                # concatenate otput array with the new components
                if i == 0:
                    grouped_output.append(copy.deepcopy(mssa.grouped_components_[j]))
                else:
                    #print("PRE  grouped_output[",j,"].shape = ",grouped_output[j].shape)
                    grouped_output[j] = np.concatenate((grouped_output[j], copy.deepcopy(mssa.grouped_components_[j])), axis = 0)
                    #print("POST grouped_output[",j,"].shape = ",grouped_output[j].shape)
                # save the correlation matrix only for the first segment
                #if i == 0:
                    # save grouped component correlation matrix
                    #ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
                    #fig, ax = plt.subplots(figsize=(12,9))
                    #sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
                    #ax.set_title('grouped component w-correlations')
                    #fig.savefig('correlation_matrix_new_'+str(j)+'.png', dpi=200)
            

            # show progress
            progress = i*100/segments
            print("Segment: ",i,"/",segments, "     Progress: ", progress," %" )
            
    # Graficar matriz de correlaciones del primero y  agrupar aditivamente los mas correlated.
        print("Original components shape: ",output.shape)
        print("Output components[0] shape: ",grouped_output[0].shape)
        # genera gráficas para cada componente con valores agrupados
        # for the 5th and the next components, save plots containing the original and cummulative timeseries for the first data column 
        # TODO: QUITAR CUANDO DE HAGA PARA TODO SEGMENTO EN EL DATASET; NO SOLO EL PRIMERO
        cumulative_recon = np.zeros_like(s_data[:, 0])
        
        # TODO : QUITAR: TEST de tamaño de grouped_components_ dictionary
        #print("len(mssa.grouped_components_) = ", str(len(mssa.grouped_components_)))
        #print("mssa.grouped_components_ = ", str(mssa.grouped_components_))
        for comp in range(len(grouped_output[0][0])):
            fig, ax = plt.subplots(figsize=(18, 7))
            current_component = grouped_output[0][:, comp]
            #print("len(grouped_output) = ", len(grouped_output))
            #print("grouped_output[0].shape = ", grouped_output[0].shape)
            
            cumulative_recon = cumulative_recon + current_component
            ax.plot(s_data[:, 0], lw=3, alpha=0.2, c='k', label='original')
            ax.plot(cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
            ax.plot(current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
            ax.legend()
            fig.savefig('mssa_' + str(comp) + '.png', dpi=600)
