# -*- coding: utf-8 -*-
"""
This File contains the MSSADecomposer class. To run this script uncomment or add the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
        mssa_decomposer = mssa_decomposer.__main__:main

Then run `python setup.py install` which will install the command `mssa_decomposer`
inside your current environment.

"""

import argparse
import sys
import logging
import numpy as np
from sklearn import preprocessing
from feature_eng.feature_eng import FeatureEng
from itertools import zip_longest 
from joblib import dump, load

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class MSSADecomposer(FeatureEng):
    """ The MSSADecomposer feature_eng class """

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
            description="Dataset MSSADecomposer: standarizes a dataset."
        )
        parser.add_argument("--no_config",
            help="Do not generate an output configuration file.",
            action="store_true",
            default=False
        )
        parser = self.parse_cmd(parser)
        pargs = parser.parse_args(args)
        self.assign_arguments(pargs)
        if hasattr(pargs, "no_config"):
            self.no_config = pargs.no_config
        else:
            self.no_config = False

    def core(self):
        """ Core feature_eng task after starting the instance with the main method.
            Decide from the arguments, what trimming method to call.

        Args:
        args (obj): command line parameters as objects
        """
        if hasattr(self, "input_config_file"):
            if self.input_config_file != None:
                self.load_from_config()
            else:
                self.standardize()
        else:
            self.standardize()
        
    def standardize(self):
        """ Standardize the dataset. """
        pt = preprocessing.StandardScaler()
        pt.fit(self.input_ds) 
        self.output_ds = pt.transform(self.input_ds) 
        dump(pt, self.output_config_file)

    def load_from_config(self):
        """ Standardize the dataset from a config file. """
        pt = preprocessing.StandardScaler()
        pt = load(self.input_config_file)
        self.output_ds = pt.transform(self.input_ds)
        
    def store(self):
        """ Save preprocessed data and the configuration of the feature_eng. """
        _logger.debug("output_file = "+ self.output_file)
        np.savetxt(self.output_file, self.output_ds, delimiter=",", fmt='%1.6f')

def run(args):
    """ Entry point for console_scripts """
    mssa_decomposer = MSSADecomposer(None)
    mssa_decomposer.main(args)


if __name__ == "__main__":
    run(sys.argv)
