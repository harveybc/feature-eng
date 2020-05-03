# -*- coding: utf-8 -*-
"""
This File contains the HeuristicTS class plugin. To run this script uncomment or add the following lines in the
[options.entry_points] section in setup.cfg:

#  TODO: MODIFICAR PARA USO COMO PLUGIN

    console_scripts =
        heu = heuristic_ts.__main__:main

Then run `python setup.py install` which will install the command `data-trimmer`
inside your current environment.

Non-Related Personal Note:

This module is dedicated to my little kitty Palomita, today 2020/04/26 may her soul rest in peace.

The whole Singularity project was inspired in your eyes, what they made me feel, made me think that 
there is much more in life than we can understand. The peace of mind i used to feel when i saw your eyes changed me forever. 

Pequeña Palomita linda, mi amor, my soul is binded to yours since we shared our lives, i loved you so much since you 
were a baby until you were old, i gave you millions of kisses and you rubbed your ears on me countles times, 
i enjoyed it every time, and your memories have influenced and will influence every step i made for good.

I miss you so much! Thanks for everything! You teached me a lot on how to love.

You were my company, my inspiration and my rock in a lonely world that does not believe in me or what i do.

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


class HeuristicTS(FeatureEng): 
    """ The Data Trimmer feature_eng class """

    def __init__(self, conf):
        """ Constructor using same parameters as base class """
        if conf != None:
            self.input_file = conf.input_file
            """ Path of the input dataset """
            if hasattr(conf, "output_file"):
                self.output_file = conf.output_file
            else:
                self.output_file = self.input_file + ".output"
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
            if hasattr(conf, "ema_fast"):
                self.ema_fast = conf.ema_fast
            else:
                self.ema_fast = 0
            if hasattr(conf, "ema_slow"):
                self.ema_slow = conf.ema_slow
            else:
                self.ema_slow = 1
            if hasattr(conf, "forward_ticks"):
                self.forward_ticks = conf.forward_ticks
            else:
                self.forward_ticks = 10
            if hasattr(conf, "plugin"):
                self.plugin = conf.plugin
            else:
                self.plugin = None
            if hasattr(conf, "list_plugins"):
                    self.list_plugins = True
            else:
                self.list_plugins = False
        else:
            self.input_ds = None
        self.r_rows = []
        self.r_cols = []
        self.config_ds = None

        
    def parse_args(self, args):
        """ Parse command line parameters

        Args:
            args ([str]): command line parameters as list of strings

        Returns:
            :obj:`argparse.Namespace`: command line parameters namespace
        """
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

