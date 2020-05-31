# -*- coding: utf-8 -*-
"""
This File contains the MSSAPredictor class plugin.
"""

from feature_eng.plugin_base import PluginBase
import numpy as np
from sys import exit
from pymssa import MSSA
import copy
import json
import seaborn as sns
import matplotlib.pyplot as plt

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class MSSAPredictor(PluginBase):
    """ Core plugin for the FeatureEng class, after initialization, saves the data and after calling the store_data method """

    def __init__(self, conf):
        """ Initializes PluginBase. Do NOT delete the following line whether you have initialization code or not. """
        super().__init__(conf)
        # Insert your plugin initialization code here.
        pass

    def parse_cmd(self, parser):
        """ Adds command-line arguments to be parsed, overrides base class """
        parser.add_argument("--num_components", help="Number of SSA components per input feature. Defaults to 0 = Autocalculated usign Singular Value Hard Thresholding (SVHT).", default=0, type=int)
        parser.add_argument("--window_size", help="Size of the data window that is half of the segments in which the dataset will be divided for analysis.", default=30, type=int)
        parser.add_argument("--forward_ticks", help="Number of ticks in the future to predict.", default=10, type=int)
        parser.add_argument("--plot_prefix", help="Exports plots of each grouped channel superposed to the input dataset. Defaults to None.", default=None, type=str)
        parser.add_argument("--show_error", help="Calculate the Mean Squared Error (MSE) between the prediction and the input future value. Defaults to False", action="store_true", default=False, type=bool)
        return parser

#TODO: SLIDING WINDOW Y PREDICCION
    def core(self, input_ds):
        """ Performs sliding-window mssa_decomposition and prediction of each input feature. """
        # get the size of the input dataset, try if there are more than one column, else, assign number of columns as 1
        try:
            self.rows_d, self.cols_d = input_ds.shape
        except:
            (self.rows_d,) = input_ds.shape
            self.cols_d = 1
            input_ds = input_ds.reshape(self.rows_d, self.cols_d)
        # create an empty array with the estimated output shape
        self.output_ds = np.empty(shape=(self.rows_d-(self.conf.window_size), self.cols_d))
        
        # center the input_ds before fitting
        in_means = np.nanmean(input_ds, axis=0)
        input_ds = input_ds - in_means

        # calculate the output by performing MSSA on <segments> number of windows of data of size window_size
        segments = (self.rows_d - (2*self.conf.window_size + self.conf.forward_ticks))
        grouped_output = []
        for i in range(0, segments):
            #progress = i*100/segments
            #print("Segment: ",i,"/",segments, "     Progress: ", progress," %" )
            # verify if i+(2*self.conf.window_size) is the last observation
            first = i 
            if (i != segments-1):
                last = i + (2 * self.conf.window_size)
            else:
                last = self.rows_d
            # slice the input_ds dataset in 2*self.conf.window_size ticks segments
            s_data_w = input_ds[first : last,:]
            # center the data before fitting
            # only the first time, run svht, in following iterations, use the same n_components, without executing the svht algo
            if i == 0:
                # uses SVHT for selecting number of components if required from the conf parameters
                if self.conf.num_components == 0:
                    mssa = MSSA(n_components='svht', window_size=self.conf.window_size, verbose=False)
                    mssa.fit(s_data_w)
                    print("Automatically Selected Rank (number of components)= ",str(mssa.rank_))
                    rank = int(mssa.rank_)
                else:
                    rank = self.conf.num_components
                    mssa = MSSA(n_components=rank, window_size=self.conf.window_size, verbose=False)
                    mssa.fit(s_data_w)
            else:
                mssa = MSSA(n_components=rank, window_size=self.conf.window_size, verbose=False)
                mssa.fit(s_data_w)

            # TODO : Con las componentes, generar la predicción y luego los plots para cada feature del input_ds
            fc = mssa.forecast(self.conf.forward_ticks, timeseries_indices=None)        
            # extracts the required tick from prediction for each feature in fc_col
            fc_col = fc[:,self.conf.forward_ticks-1]
            (rows_o,) = fc_col.shape
            # transpose the predictions into a row 
            fc_row = fc_col.reshape(1,rows_o)
            # extract the row of components for all features
            print("mssa.components_[:, i + (2 * self.conf.window_size) -1 , :].shape = ", mssa.components_[:, i + (2 * self.conf.window_size) -1 , :].shape)
            comp_row = mssa.components_[:, i + (2 * self.conf.window_size) -1 , :].sum(axis=1)
            print("comp_row.shape = ", comp_row.shape)

            # TODO: concatenate otput array with the new predictions
            if i == 0:
                self.output_ds = fc_row
                denoised = np.array(mssa.components_)
                
                print("ini self.output_ds.shape = ", self.output_ds.shape)
                
            else:
                self.output_ds = np.concatenate((self.output_ds, fc_row), axis = 0)
            # TODO: calculate error per feature
    

        print("end self.output_ds.shape = ", self.output_ds.shape)
        if self.conf.plot_prefix != None:
            # Graficar matriz de correlaciones del primero y  agrupar aditivamente los mas correlated.
            # genera gráficas para cada componente con valores agrupados
            # for the 5th and the next components, save plots containing the original and cummulative timeseries for the first data column
            # TODO: QUITAR CUANDO DE HAGA PARA TODO SEGMENTO EN EL DATASET; NO SOLO EL PRIMERO
            cumulative_recon = np.zeros_like(input_ds[:, 0])
            # TODO : QUITAR: TEST de tamaño de grouped_components_ dictionary
            for comp in range(len(grouped_output[0][0])):
                fig, ax = plt.subplots(figsize=(18, 7))
                current_component = self.output_ds[0,:, comp]
                cumulative_recon = cumulative_recon + current_component
                ax.plot(input_ds[:, 0], lw=3, alpha=0.2, c='k', label='original')
                ax.plot(cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
                ax.plot(current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
                ax.legend()
                fig.savefig(self.conf.plot_prefix + '_' + str(comp) + '.png', dpi=600)

        print("new self.output_ds.shape = ", self.output_ds.shape)
        return self.output_ds