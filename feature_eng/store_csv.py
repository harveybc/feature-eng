# -*- coding: utf-8 -*-
"""
This File contains the StoreCSV class plugin. 
"""

from numpy import savetxt
from sys import exit

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class StoreCSV(): 
    """ Output plugin for the FeatureEng class, after initialization, saves the data and after calling the store_data method """

    def __init__(self, conf):
        """ Constructor using same parameters as base class """
        if conf != None:
            self.assign_arguments(conf)
    
    def store_csv(self, output_ds):
        """ Save preprocessed data """
        savetxt(self.output_file, output_ds, delimiter=",")
        
    def assign_arguments(self,conf):
        """ Assign configuration values to class attributes""" 
        if hasattr(conf, "input_file"):
            self.input_file = conf.input_file
        if hasattr(conf, "output_file"):
            self.output_file = conf.output_file
        else:
            print("Warning: No output_file parameter provided for store_csv plugin. Using default input_file with .output extension.")
            self.output_file = conf.input_file + ".output"
            
    