# -*- coding: utf-8 -*-
""" This File contains the FeatureEng class, it is the base class for HeuristicTS, FeatureSelector, Standardizer and SlidingWindow classes. """

import argparse
import sys
import logging
import numpy as np
import csv
from feature_eng.feature_eng_base import FeatureEngBase

# from feature_eng import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class FeatureEng(FeatureEngBase):
    """ Base class for HeuristicTS, FeatureSelector, Standardizer, SlidingWindow. """

    def main(self, args):
        """ Starts an instance. Main entry point allowing external calls.
            Starts logging, parse command line arguments and start core.

        Args:
        args ([str]): command line parameter list
        """
        self.parse_args(args)
        # Start logging: TODO: Use args.loglevel en lugar de logging.DEBUG
        self.setup_logging(logging.DEBUG)
        _logger.info("Starting feature_eng...")
        # Load input dataset
        if self.input_ds == None:
            _logger.debug("Loading input file.")
            self.load_ds()
        # Start core function
        self.core()
        # Start logger
        _logger.debug("Saving results.")
        # Save results and output configuration
        self.store()
        _logger.info("Script end.")

    def parse_cmd(self, parser):
        parser.add_argument("--version", action="version", version="feature_eng")
        parser.add_argument("--input_file", help="Input CSV filename ")
        parser.add_argument("--output_file", help="Output CSV filename")
        parser.add_argument("--input_config_file", help="Input configuration  filename")
        parser.add_argument("--output_config_file", help="Output configuration  filename")
        parser.add_argument("-v","--verbose",dest="loglevel",help="set loglevel to INFO",action="store_const",const=logging.INFO)
        parser.add_argument("-vv","--very_verbose",dest="loglevel",help="set loglevel to DEBUG",action="store_const",const=logging.DEBUG)
        return parser
    
    def assign_arguments(self,pargs):
        if hasattr(pargs, "input_file"):
            if pargs.input_file != None: 
                self.input_file = pargs.input_file
                if hasattr(pargs, "output_file"):
                    if pargs.output_file != None: self.output_file = pargs.output_file
                    else: self.output_file = self.input_file + ".output"
                else:
                    self.output_file = self.input_file + ".output"
                if hasattr(pargs, "input_config_file"):
                    if pargs.input_config_file != None: self.input_config_file = pargs.input_config_file
                    else: self.input_config_file = None
                else:
                    self.input_config_file = None
                if hasattr(pargs, "output_config_file"):
                    if pargs.output_config_file != None: self.output_config_file = pargs.output_config_file
                    else: self.output_config_file = self.input_file + ".config" 
                else:
                    self.output_config_file = self.input_file + ".config"
            else:
                print("Error: No input file parameter provided. Use option -h to show help.")
                sys.exit()
        else:
            print("Error: No input file parameter provided. Use option -h to show help.")
            sys.exit()

    def store(self):
        """ Save preprocessed data and the configuration of the feature_eng. """
        pass

    def core(self):
        """ Core feature_eng task after starting the instance with the main method.
            To be overriden by child classes depending on their feature_eng task.
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

