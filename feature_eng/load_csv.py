# -*- coding: utf-8 -*-
"""
This File contains the LoadCSV class plugin. To run this script uncomment or add the following lines in the

"""

import argparse
import sys
import logging
import numpy as np
from feature_eng.feature_eng import FeatureEng
from itertools import zip_longest 

# from heuristic_ts import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class LoadCSV(FeatureEngBase): 
    """ input plugin for the FeatureEng class """

    def __init__(self, conf):
        """ Constructor using same parameters as base class """
        if conf != None:
            self.assign_arguments(conf)
    
    def load_data(self):
        """ Load the input dataset """
        # Load input dataset
        self.input_ds = np.genfromtxt(self.input_file, delimiter=",")
        # load input config dataset if the parameter is available
        return self.input_ds
        
    def parse_args(self, args):
        """ Parse command line parameters        """
        parser = argparse.ArgumentParser(
            description="Dataset Trimmer: trims constant columns and consecutive zero rows from the end and the start of a dataset."
        )
        parser.add_argument(
            "--ema_fast",
            help="column index on the input dataset for ema fast",
            type=int,
            default=0
        )
        parser.add_argument("--ema_slow",
            help="column index on the input dataset for ema slow",
            type=int,
            default=0
        )
        parser.add_argument("--forward_ticks",
            help="number of ticks in the future for ema_fast",
            type=int,
            default=0
        )
        parser = self.parse_cmd(parser)
        pargs = parser.parse_args(args)
        self.assign_arguments(pargs)
        if hasattr(pargs, "ema_fast"):
            self.ema_fast = pargs.ema_fast
        else:
            self.ema_fast = 0
        if hasattr(pargs, "ema_slow"):
            self.ema_slow = pargs.ema_slow
        else:
            self.ema_slow = 1
        if hasattr(pargs, "forward_ticks"):
            self.forward_ticks = pargs.forward_ticks
        else:
            self.forward_ticks = 10
 