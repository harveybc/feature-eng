# -*- coding: utf-8 -*-
"""
This File contains the StoreCSV class plugin. 
"""

import numpy as np
from sys import exit

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class HeuristicTS(): 
    """ Output plugin for the FeatureEng class, after initialization, saves the data and after calling the store_data method """

    def __init__(self, conf):
        """ Constructor using same parameters as base class """
        if conf != None:
            self.assign_arguments(conf)

    def core(self, input_ds):
        """ Performs the substraction of the ema_fast forwarded forward_ticks
            minus the ema_slow.
        """
        # get the size of the input dataset
        self.rows_d, self.cols_d = input_ds.shape
        # create an empty array with the estimated output shape
        self.output_ds = np.empty(shape=(self.rows_d-self.forward_ticks, 1))
        # calculate the output
        for i in range(self.rows_d - self.forward_ticks): 
            self.output_ds[i] = self.input_ds[i+self.forward_ticks, self.ema_fast]-self.input_ds[i, self.ema_slow]
        return self.output_ds

    def assign_arguments(self,conf):
        """ Assign configuration values to class attributes""" 
        if hasattr(conf, "forward_ticks"):
            self.forward_ticks = conf.forward_ticks
        else:
            self.forward_ticks = 10
        if hasattr(conf, "ema_fast"):
            self.ema_fast = conf.ema_fast
        else:
            self.ema_fast = 0
        if hasattr(conf, "ema_slow"):
            self.ema_slow = conf.ema_slow
        else:
            self.ema_slow = 1
            
    