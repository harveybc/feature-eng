# -*- coding: utf-8 -*-
"""
This File contains the LoadCSV class plugin. 
"""

from numpy import genfromtxt
from sys import exit

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class LoadCSV(): 
    """ input plugin for the FeatureEng class, after initialization, the input_ds attribute is set """

    def __init__(self, conf):
        """ Constructor using same parameters as base class """
        if conf != None:
            self.assign_arguments(conf)
    
    def load_data(self):
        """ Load the input dataset """
        # Load input dataset
        self.input_ds = genfromtxt(self.input_file, delimiter=",")
        return self.input_ds
        
    def assign_arguments(self,conf):
        """ Assign configuration values to class attributes""" 
        if hasattr(conf, "input_file"):
            self.input_file = conf.input_file
        else:
            print("Error: No input_file parameter provided for load_csv plugin.")
            exit()
            
    