# -*- coding: utf-8 -*-
"""
This File contains the StoreCSV class plugin. 
"""

from feature_eng.plugin_base import PluginBase
import numpy as np
from sys import exit

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class HeuristicTS(PluginBase): 
    """ Output plugin for the FeatureEng class, after initialization, saves the data and after calling the store_data method """

    def __init__(self, conf):
        """ Initializes PluginBase. Do NOT delete the following line whether you have initialization code or not. """
        super().__init__(conf)
        # Insert your plugin initialization code here.
        pass

    def parse_cmd(self, parser):
        """ Adds command-line arguments to be parsed, overrides base class """
        parser.add_argument("--forward_ticks", help="Number of forwrard ticks in the future for ema_fast", default=10)
        parser.add_argument("--ema_fast", help="Column index for ema fast", default=0)
        parser.add_argument("--ema_slow", help="Column index for ema slow", default=1)
        return parser

    def core(self, input_ds):
        """ Performs the substraction of the ema_fast forwarded forward_ticks
            minus the ema_slow.
        """
        # get the size of the input dataset
        self.rows_d, self.cols_d = input_ds.shape
        # create an empty array with the estimated output shape
        self.output_ds = np.empty(shape=(self.rows_d-self.conf.forward_ticks, 1))
        # calculate the output
        for i in range(self.rows_d - self.conf.forward_ticks): 
            self.output_ds[i] = input_ds[i+self.conf.forward_ticks, self.conf.ema_fast]-input_ds[i, self.conf.ema_slow]
        return self.output_ds
