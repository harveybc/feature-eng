# -*- coding: utf-8 -*-
""" This File contains the FeatureEng class, it is the base class for HeuristicTS, FeatureSelector, Standardizer, SlidingWindow. """

import argparse
import sys
import logging
import numpy as np
import csv

# from feature_eng import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)

class FeatureEngBase:
    """ Base class for FeatureEng. """

    def __init__(self, conf):
        """ Constructor """
        # if conf =  None, loads the configuration from the command line arguments
        if conf != None:
            self.input_file = conf.input_file
            """ Path of the input dataset """
            self.output_file = conf.output_file
            """ Path of the output dataset """
            if hasattr(conf, "input_config_file"):
                self.input_config_file = conf.input_config_file
            else:
                self.input_config_file = None
            """ Path of the input configuration """
            if hasattr(conf, "output_config_file"):
                self.output_config_file = conf.output_config_file
            else:
                self.output_config_file = None
            """ Path of the output configuration """
            # Load input dataset
            self.load_ds()
        else :
            self.input_ds = None
        self.r_rows = []
        self.r_cols = []
        self.config_ds = None

    def setup_logging(self, loglevel):
        """Setup basic logging.

        Args:
        loglevel (int): minimum loglevel for emitting messages
        """
        logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
        logging.basicConfig(
            level=loglevel,
            stream=sys.stdout,
            format=logformat,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def load_ds(self):
        """ Save preprocessed data and the configuration of the feature_eng. """
        # Load input dataset
        self.input_ds = np.genfromtxt(self.input_file, delimiter=",")
        # load input config dataset if the parameter is available
        # Initialize input number of rows and columns
        self.rows_d, self.cols_d = self.input_ds.shape
    
    