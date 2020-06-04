# -*- coding: utf-8 -*-
""" This File contains the FeatureEng class, has methods for listing and loading plugins and execute their entry point. """

import argparse
import sys
import logging
import numpy as np
import csv
import pkg_resources

# from feature_eng import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)

class FeatureEngBase():
    """ Base class For FeatureEng. """
    
    def __init__(self, conf):
        """ Constructor """
        self.conf = conf
        if conf != None:
            if not hasattr(conf, "args"):
                self.conf.args = None
                self.setup_logging(logging.DEBUG)
                _logger.info("Starting feature_eng via class constructor...")
                # list available plugins
                if self.conf.list_plugins == True:
                    _logger.debug("Listing plugins.")
                    self.find_plugins()
                    _logger.debug("Printing plugins.")
                    self.print_plugins()
                # execute core operations
                else: 
                    # sets default values for plugins
                    if not hasattr(conf, "input_plugin"): 
                        self.conf.input_plugin = "load_csv"    
                    if not hasattr(conf, "output_plugin"): 
                        self.conf.output_plugin = "store_csv"
                    if not hasattr(conf, "core_plugin"): 
                        self.conf.core_plugin = "heuristic_ts"
                    self.core()

    def parse_cmd(self, parser):
        """ Adds command-line arguments to parse """
        parser.add_argument("--version", action="version", version="feature_eng")
        parser.add_argument("--list_plugins", help="lists all installed external and internal plugins", default=False)
        parser.add_argument("--core_plugin", help="Plugin to load ", default="heuristic_ts")
        parser.add_argument("--input_plugin", help="Input plugin to load ", default="load_csv")
        parser.add_argument("--output_plugin", help="Output plugin to load", default="store_csv")
        parser.add_argument("-v","--verbose",dest="loglevel",help="set loglevel to INFO",action="store_const",const=logging.INFO)
        parser.add_argument("-vv","--very_verbose",dest="loglevel",help="set loglevel to DEBUG",action="store_const",const=logging.DEBUG)
        return parser
    
    def core(self):
        """ Core feature_eng operations. """
        _logger.debug("Finding Plugins.")
        self.find_plugins()
        _logger.debug("Loading plugins.")
        self.load_plugins()
        _logger.debug("Loading input dataset from the input plugin.")
        self.input_ds = self.ep_input.load_data() 
        _logger.debug("Performing core operations from the  core plugin.")
        self.output_ds = self.ep_core.core(self.input_ds) 
        _logger.debug("Storing results using the output plugin.")
        self.ep_output.store_data(self.output_ds) 
        _logger.info("feature_eng finished.")
    
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
    
    def parse_args(self, args):
        """Parse command line parameters.

        Args:
        args ([str]): command line parameters as list of strings

        Returns:
        :obj:`argparse.Namespace`: command line parameters namespace
        """
        parser = argparse.ArgumentParser(
            description="FeatureEng: Feature engineering operations."
        )
        parser = self.parse_cmd(parser)
        self.conf, self.unknown = parser.parse_known_args(args)
        # assign as arguments, the unknown arguments from the parser
        self.conf.args = self.unknown


        