# -*- coding: utf-8 -*-
""" This File contains the FeatureEng class, has methods for listing and loading plugins and execute their entry point. """

import argparse
import sys
import logging
import numpy as np
import csv
import pkg_resources
from feature_eng.feature_eng_base import FeatureEngBase


# from feature_eng import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class FeatureEng(FeatureEngBase):
    """ Base class. """

    def __init__(self, conf):
        """ Constructor """
        # if conf =  None, loads the configuration from the command line arguments
        if conf != None:
            self.setup_logging(logging.DEBUG)
            _logger.info("Starting feature_eng via class constructor...")
            # assign arguments to class attributes
            self.conf =  conf
            self.assign_arguments(conf)
            # list available plugins
            if hasattr(self, "list_plugins"):
                _logger.debug("Listing plugins.")
                self.find_plugins()
                _logger.debug("Printing plugins.")
                self.print_plugins()
            # execute core operations
            elif hasattr(self, "core_plugin"):
                self.core()
            
    def main(self, args):
        """ Starts an instance. Main entry point allowing external calls.
            Starts logging, parse command line arguments and start core.

        Args:
        args ([str]): command line parameter list
        """
        self.setup_logging(logging.DEBUG)
        self.parse_args(args)
        if self.core_plugin != None:    
            self.core(self.unknown)
        else:
            if self.list_plugins == True:
                _logger.debug("Listing plugins.")
                self.find_plugins()
                _logger.debug("Printing plugins.")
                self.print_plugins()
            else: 
                _logger.debug("Error: No core plugin provided. for help, use feature_eng --help")
        _logger.info("Script end.")

    def load_plugins(self):
        """ Loads plugin entry points into class attributes"""
        if self.input_plugin in self.discovered_input_plugins:
            self.ep_i = self.discovered_input_plugins[self.input_plugin]
            self.ep_input = self.ep_i(self.conf)
        else:
            print("Error: Input Plugin "+ self.input_plugin +" not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()
        if self.output_plugin in self.discovered_output_plugins:
            self.ep_o = self.discovered_output_plugins[self.output_plugin]
            self.ep_output = self.ep_o(self.conf)
        else:
            print("Error: Output Plugin "+ self.output_plugin +" not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()
        if self.core_plugin in self.discovered_core_plugins:
            self.ep_c = self.discovered_core_plugins[self.core_plugin]
            self.ep_core = self.ep_c(self.conf)
        else:
            print("Error: Core Plugin "+ self.core_plugin +" not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()
    
    def find_plugins(self):
        """" Populate the discovered plugin lists """
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
        
def run(args):
    """ Entry point for console_scripts """
    feature_eng = FeatureEng(None)
    feature_eng.main(args)


if __name__ == "__main__":
    run(sys.argv)
