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
    """ Base class For FeatureEng and its plugins. """

    def __init__(self, conf):
        """ Constructor """
        # if conf =  None, loads the configuration from the command line arguments
        if conf != None:
            # assign arguments to class attributes
            self.assign_arguments(conf)  
            
    def parse_cmd(self, parser):
        """ Adds command-line arguments to parse """
        parser.add_argument("--version", action="version", version="feature_eng")
        parser.add_argument("--list_plugins", help="lists all installed external and internal plugins")
        parser.add_argument("--core_plugin", help="Plugin to load ")
        parser.add_argument("--input_plugin", help="Input CSV filename ")
        parser.add_argument("--output_plugin", help="Output CSV filename")
        parser.add_argument("-v","--verbose",dest="loglevel",help="set loglevel to INFO",action="store_const",const=logging.INFO)
        parser.add_argument("-vv","--very_verbose",dest="loglevel",help="set loglevel to DEBUG",action="store_const",const=logging.DEBUG)
        return parser
    
    def assign_arguments(self,conf):
        """ Assign configuration values to class attributes""" 
        if hasattr(conf, "core_plugin"):
            self.core_plugin = conf.core_plugin
            if hasattr(conf, "input_plugin"):
                self.input_plugin = conf.input_plugin
            else:
                self.input_plugin = "load_csv"
            if hasattr(conf, "output_plugin"):
                self.output_plugin = conf.output_plugin
            else:
                self.output_plugin = "store_csv"
        else:
            if hasattr(conf, "list_plugins"):
                if conf.list_plugins == True:
                    self.list_plugins = True
                else:
                    print("Error: No valid parameters provided. Use option -h to show help.")
                    sys.exit()
        
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
        """Parse command line parameters, to be overriden by child classes depending on their command line parameters if they are console scripts.

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
        self.assign_arguments(self.conf)
        


