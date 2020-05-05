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


class FeatureEng():
    """ Base class. """

    def __init__(self, conf):
        """ Constructor """
        # if conf =  None, loads the configuration from the command line arguments
        if conf != None:
            self.setup_logging(logging.DEBUG)
            _logger.info("Starting feature_eng via class constructor...")
            # assign arguments to class attributes
            self.assign_arguments(conf)
            # execute core operations
            if hasattr(self, "core_plugin"):
                self.core(conf)
            # list available plugins
            elif hasattr(self, "list_plugins"):
                _logger.debug("Listing plugins.")
                self.find_plugins()
                _logger.debug("Printing plugins.")
                self.print_plugins()
            
    def parse_cmd(self, parser):
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
                self.input_plugin = "csv_input"
            if hasattr(conf, "output_plugin"):
                self.output_plugin = conf.output_plugin
            else:
                self.output_plugin = "csv_output"
        elif hasattr(conf, "list_plugins"):
            self.list_plugins = conf.list_plugins
        else:
            print("Error: No valid parameters provided. Use option -h to show help.")
            sys.exit()
        

    def main(self, args):
        """ Starts an instance. Main entry point allowing external calls.
            Starts logging, parse command line arguments and start core.

        Args:
        args ([str]): command line parameter list
        """
        self.setup_logging(logging.DEBUG)
        self.parse_args(args)
        if self.plugin != None:    
            self.core(conf)
        else:
            if self.list_plugins == True:
                self.find_plugins()
            else: 
                _logger.debug("Error: No core plugi provided. for help, use feature_eng --help")
        _logger.info("Script end.")

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

    def initialize_plugins(self):
        if self.plugin in self.discovered_plugins:
            self.plugin_entry_point = self.discovered_plugins[self.plugin]
        else:
            print("Error: Plugin "+ self.plugin +" not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()

    def load_plugins(self):
        """ Loads plugin entry points into class attributes"""
        if self.input_plugin in self.discovered_input_plugins:
            self.ep_input = self.discovered_input_plugins[self.input_plugin]
        else:
            print("Error: Input Plugin "+ self.input_plugin +" not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()
        if self.output_plugin in self.discovered_output_plugins:
            self.ep_output = self.discovered_output_plugins[self.output_plugin]
        else:
            print("Error: Output Plugin "+ self.output_plugin +" not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()
        if self.core_plugin in self.discovered_core_plugins:
            self.ep_core = self.discovered_core_plugins[self.core_plugin]
        else:
            print("Error: Core Plugin "+ self.core_plugin +" not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()
    
    def find_plugins(self):
        self.discovered_input_plugins = {
            entry_point.name: entry_point.load()
            for entry_point
            in pkg_resources.iter_entry_points('feature_eng.plugins.input')
        }
        self.discovered_output_plugins = {
            entry_point.name: entry_point.load()
            for entry_point
            in pkg_resources.iter_entry_points('feature_eng.plugins.output')
        }
        self.discovered_core_plugins = {
            entry_point.name: entry_point.load()
            for entry_point
            in pkg_resources.iter_entry_points('feature_eng.plugins.core')
        }

    def print_plugins(self):
        print("Discovered input plugins:")
        for key in self.discovered_input_plugins:
            print(key+"\n")
        print("Discovered output plugins:")
        for key in self.discovered_output_plugins:
            print(key+"\n")
        print("Discovered core plugins:")
        for key in self.discovered_core_plugins:
            print(key+"\n")

    def core(self,conf):
        """ Core feature_eng operations. """
        self.core_plugin = conf.core_plugin
        """ Core plugin to load """
        _logger.debug("Finding Plugins.")
        self.find_plugins()
        _logger.debug("Loading plugins.")
        self.load_plugins()
        _logger.debug("Initializing plugins.")
        self.init_plugins(conf)
        _logger.debug("Loading input dataset from the input plugin.")
        self.input_ds = self.ep_input.load() 
        _logger.debug("Performing core operations from the  core plugin.")
        self.output_ds = self.ep_core.core(self.input_ds) 
        _logger.debug("Storing results using the output plugin.")
        self.ep_output.store(self.output_ds) 
        _logger.info("feature_eng finished.")
            
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
        conf, unknown = parser.parse_known_args(args)
        self.assign_arguments(conf)
        
def run(args):
    """ Entry point for console_scripts """
    feature_eng = FeatureEng(None)
    feature_eng.main(args)


if __name__ == "__main__":
    run(sys.argv)
