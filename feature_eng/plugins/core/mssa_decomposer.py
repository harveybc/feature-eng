# -*- coding: utf-8 -*-
"""
This File contains the MSSADecomposer class plugin.
"""

from feature_eng.plugin_base import PluginBase
import numpy as np
from sys import exit
from pymssa import MSSA
import copy
import json
import seaborn as sns


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
        parser.add_argument("--num_components", help="Number of SSA components per input feature. Defaults to 0 = Autocalculated usign Singular Value Hard Thresholding (SVHT).", default=0, type=int)
        parser.add_argument("--window_size", help="Size of the data windows in which the dataset will be divided for analysis.", default=30, type=int)
        parser.add_argument("--group_file", help="Filename for the JSON file containing manually set feature groups. Use --plot_correlation to export a w-correlation matrix plot. Defaults to None.", default="test", type=str)
        parser.add_argument("--w_prefix", help="Exports a plot of the w-correlation matrix for grouped components. Defaults to None.", default=None, type=str)
        parser.add_argument("--plot_prefix", help="Exports plots of each grouped channel superposed to the input dataset. Defaults to None.", default=30, type=str)
        return parser

    def core(self, input_ds):
        """ Performs mssa_decomposition. """
        # get the size of the input dataset
        self.rows_d, self.cols_d = input_ds.shape
        # create an empty array with the estimated output shape
        self.output_ds = np.empty(shape=(self.rows_d-self.conf.window_size, 1))
        # calculate the output by performing MSSA on <segments> number of windows of data of size window_size
        segments = (self.rows_d // (2*self.conf.window_size))
        for i in range(0, segments):
            # verify if i+(2*self.conf.window_size) is the last observation
            first = i * (2 * self.conf.window_size)
            if (i != segments-1):
                last = (i+1) * (2 * self.conf.window_size)
            else:
                last = self.rows_d
            # slice the input_ds dataset in 2*self.conf.window_size ticks segments
            s_data_w = input_ds[first : last,:]
            # only the first time, run svht, in following iterations, use the same n_components, without executing the svht algo
            if i == 0:
                # uses SVHT for selecting number of components if required from the conf parameters
                if self.conf.num_components == 0:
                    mssa = MSSA(n_components='svht', window_size=self.conf.window_size, verbose=True)
                    mssa.fit(s_data_w)
                    print("Automatically Selected Rank (number of components)= ",str(mssa.rank_))
                    rank = int(mssa.rank_)
                else:
                    rank = self.conf.num_components
                    mssa = MSSA(n_components=rank, window_size=self.conf.window_size, verbose=True)
                    mssa.fit(s_data_w)
            else:
                mssa = MSSA(n_components=rank, window_size=self.conf.window_size, verbose=True)
                mssa.fit(s_data_w)

            # concatenate otput array with the new components
            if i == 0:
                self.output_ds = copy.deepcopy(mssa.components_)
                print("output_ds.shape = ", self.output_ds.shape)
            else:
                print("output_ds.shape = ", self.output_ds.shape)
                print("mssa.components_ = ", mssa.components_.shape)
                self.output_ds = np.concatenate((self.output_ds, mssa.components_), axis = 1)
                print("new output_ds.shape = ", self.output_ds.shape)
            # use the same groups for all the features
            # load the groups from a json file
            grouped_output = []
            if self.conf.group_file != None:
                #TODO: concatenate grouped output
                print("Grouping correlated components (manually set list)")
                # TODO: QUITAR GUARDADO DE JSON DE EJEMPLO
                ts0_groups = [[0],[1],[2],[3],[4,5],[6],[7],[8],[9,10],[11],[12]]
                with open(self.conf.group_file, 'w') as f:
                    json.dump(ts0_groups, f)
                with open(self.conf.group_file) as json_file:
                    ts0_groups = json.load(json_file)
                for j in range(0, self.cols_d):
                    # draw correlation matrix for the first segment
                    mssa.set_ts_component_groups(j, ts0_groups)
                    ts0_grouped = mssa.grouped_components_[j]
                    # concatenate otput array with the new components
                    if i == 0:
                        grouped_output.append(copy.deepcopy(mssa.grouped_components_[j]))
                    else:
                        grouped_output[j] = np.concatenate((grouped_output[j], copy.deepcopy(mssa.grouped_components_[j])), axis = 0)
                    # save the correlation matrix only for the first segment
                    if (i == 0) and (self.conf.plot_correlations != None):
                        # save grouped component correlation matrix
                        ts0_grouped_wcor = mssa.w_correlation(ts0_grouped)
                        fig, ax = plt.subplots(figsize=(12,9))
                        sns.heatmap(np.abs(ts0_grouped_wcor), cmap='coolwarm', ax=ax)
                        ax.set_title('grouped component w-correlations')
                        fig.savefig(self.conf.plot_correlations + str(j) + 'grouped.png', dpi=200)
                self.output_ds = grouped_output
            else:
                grouped_output = self.output_ds
        # show progress
        progress = i*100/segments
        print("Segment: ",i,"/",segments, "     Progress: ", progress," %" )
        if self.conf.plot_prefix != None:
            # Graficar matriz de correlaciones del primero y  agrupar aditivamente los mas correlated.
            # genera gráficas para cada componente con valores agrupados
            # for the 5th and the next components, save plots containing the original and cummulative timeseries for the first data column
            # TODO: QUITAR CUANDO DE HAGA PARA TODO SEGMENTO EN EL DATASET; NO SOLO EL PRIMERO
            cumulative_recon = np.zeros_like(s_data[:, 0])
            # TODO : QUITAR: TEST de tamaño de grouped_components_ dictionary
            for comp in range(len(grouped_output[0][0])):
                fig, ax = plt.subplots(figsize=(18, 7))
                current_component = grouped_output[0][:, comp]
                cumulative_recon = cumulative_recon + current_component
                ax.plot(s_data[:, 0], lw=3, alpha=0.2, c='k', label='original')
                ax.plot(cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
                ax.plot(current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
                ax.legend()
                fig.savefig(self.conf.plot_prefix + '_' + str(comp) + '.png', dpi=600)
        print("pre self.output_ds.shape = ", self.output_ds.shape)

        # transforms the dimensions from (features, ticks, channels) to (ticks, feats*channels)
        ns_output = []
        for n in range(self.output_ds.shape[1]):
            row = []
            for p in range(self.output_ds.shape[0]):
                # TODO: CORREGIR PARA CUANDO SE USE GROUP_FILE
                for c in range (self.output_ds.shape[2]):
                    #row.append(self.output_ds[p,n,c])
                    row.append(self.output_ds[p,n,c])
            ns_output.append(row)
        # convert to np array
        self.output_ds = np.array(ns_output)
        print("new self.output_ds.shape = ", self.output_ds.shape)
        return self.output_ds