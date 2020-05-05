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
            

    def core(self):
        """ Core feature_eng task after starting the instance with the main method.
            Decide from the arguments, what trimming method to call.

        Args:
        args (obj): command line parameters as objects
        """

        self.training_signal()
        
    
    def training_signal(self):
        """ Performs the substraction of the ema_fast forwarded forward_ticks
            minus the ema_slow.
        """
        self.output_ds = np.empty(shape=(self.rows_d-self.forward_ticks, 1))
        for i in range(self.rows_d - self.forward_ticks): 
            self.output_ds[i] = self.input_ds[i+self.forward_ticks, self.ema_fast]-self.input_ds[i, self.ema_slow]


    def store(self):
        """ Save preprocessed data and the configuration of the feature_eng. """
        print("self.output_ds.shape = ", self.output_ds.shape)
        _logger.debug("output_file = "+ self.output_file)
        np.savetxt(self.output_file, self.output_ds, delimiter=",")
      

def run(args):
    """ Entry point for console_scripts """
    heuristic_ts = HeuristicTS(None)
    heuristic_ts.main(args)


if __name__ == "__main__":
    run(sys.argv)

