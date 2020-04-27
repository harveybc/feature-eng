# -*- coding: utf-8 -*-
"""
This File contains the HeuristicTS class. To run this script uncomment or add the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
        data-trimmer = heuristic_ts.__main__:main

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
from feature_engineering.feature_engineering import FeatureEng
from itertools import zip_longest 

# from heuristic_ts import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class HeuristicTS(FeatureEng):
    """ The Data Trimmer feature_engineering class """

    def __init__(self, conf):
        """ Constructor using same parameters as base class """
        super().__init__(conf)

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
            "--from_start",
            help="number of rows to remove from start (ignored if auto_trim)",
            type=int,
            default=0
        )
        parser.add_argument("--from_end",
            help="number of rows to remove from end (ignored if auto_trim)",
            type=int,
            default=0
        )
        parser.add_argument("--remove_columns", 
            help="removes constant columns", 
            action="store_true",
            default=False
        )
        parser.add_argument("--no_auto_trim",
            help="trims the constant columns and trims all rows with consecutive zeroes from start and end",
            action="store_true",
            default=False
        )
        parser = self.parse_cmd(parser)
        pargs = parser.parse_args(args)
        self.assign_arguments(pargs)
        if hasattr(pargs, "from_start"):
            self.from_start = pargs.from_start
        if hasattr(pargs, "from_end"):
            self.from_end = pargs.from_end
        if hasattr(pargs, "remove_columns"):
            self.remove_columns = pargs.remove_columns
        if hasattr(pargs, "no_auto_trim"):
            self.auto_trim = not(pargs.no_auto_trim)
        else:
            self.auto_trim = True

    def core(self):
        """ Core feature_engineering task after starting the instance with the main method.
            Decide from the arguments, what trimming method to call.

        Args:
        args (obj): command line parameters as objects
        """
        self.training_signal()
        
    
    def training_signal(self):
        """ Trims all the constant columns and trims all rows with consecutive zeroes from start and end of the input dataset

        Returns:
        rows_t, cols_t (int,int): number of rows and columns trimmed
        """
        
        # TODO:  CORE 

        return rows_t, cols_t


    def store(self):
        """ Save preprocessed data and the configuration of the feature_engineering. """
        print("self.output_ds.shape = ", self.output_ds.shape)
        _logger.debug("output_file = "+ self.output_file)
        np.savetxt(self.output_file, self.output_ds, delimiter=",")
      

def run(args):
    """ Entry point for console_scripts """
    heuristic_ts = HeuristicTS(None)
    heuristic_ts.main(args)


if __name__ == "__main__":
    run(sys.argv)
