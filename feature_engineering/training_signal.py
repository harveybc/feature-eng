# -*- coding: utf-8 -*-
""" This File contains the Preprocessor class, it is the base class for DataTrimmer, FeatureSelector, Standardizer, MSSADecomposer. """

import argparse
import sys
import logging
import numpy as np
import csv
from feature_engineering.feature_engineering_base import PreprocessorBase

# from feature_engineering import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class Preprocessor(PreprocessorBase):
    """ Base class for DataTrimmer, FeatureSelector, Standardizer, MSSADecomposer. """

    def main(self, args):
        """ Starts an instance. Main entry point allowing external calls.
            Starts logging, parse command line arguments and start core.

        Args:
        args ([str]): command line parameter list
        """
        self.parse_args(args)
        # Start logging: TODO: Use args.loglevel en lugar de logging.DEBUG
        self.setup_logging(logging.DEBUG)
        _logger.info("Starting feature_engineering...")
        # Load input dataset
        if self.input_ds == None:
            self.load_ds()
        # Start core function
        self.core()
        # Start logger
        _logger.debug("Saving results...")
        # Save results and output configuration
        self.store()
        _logger.info("Script end.")

    def parse_cmd(self, parser):
        parser.add_argument("--version", action="version", version="feature_engineering")
        parser.add_argument("--input_file", help="Input CSV filename ")
        parser.add_argument("--output_file", help="Output CSV filename")
        parser.add_argument("--input_config_file", help="Input configuration  filename")
        parser.add_argument("--output_config_file", help="Output configuration  filename")
        parser.add_argument("-v","--verbose",dest="loglevel",help="set loglevel to INFO",action="store_const",const=logging.INFO)
        parser.add_argument("-vv","--very_verbose",dest="loglevel",help="set loglevel to DEBUG",action="store_const",const=logging.DEBUG)
        return parser
    
    def store(self):
        """ Save preprocessed data and the configuration of the feature_engineering. """
        pass

    def core(self):
        """ Core feature_engineering task after starting the instance with the main method.
            To be overriden by child classes depending on their feature_engineering task.
        """
        pass

    def parse_args(self, args):
        """Parse command line parameters, to be overriden by child classes depending on their command line parameters if they are console scripts.

        Args:
        args ([str]): command line parameters as list of strings

        Returns:
        :obj:`argparse.Namespace`: command line parameters namespace
        """
        pass

