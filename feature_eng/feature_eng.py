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
        """ Initializes FeatureEngBase.  """
        super().__init__(conf)
        
    def main(self, args):
        """ Starts an instance via command line parameters, uses the FeatureEngBase.core() method.
            Starts logging, parse command line arguments and start core.

        Args:
        args ([str]): command line parameter list
        """
        self.setup_logging(logging.DEBUG) 
        self.parse_args(args)
        if self.conf.core_plugin != None:    
            self.core()
        else:
            if self.conf.list_plugins == True:
                _logger.debug("Finding plugins.")
                self.find_plugins()
                _logger.debug("Printing plugins.")
                self.print_plugins()
            else: 
                _logger.debug("Error: No core plugin provided. for help, use feature_eng --help")
        _logger.info("Script end.")

    def find_plugins(self):
        """" Populate the discovered plugin lists """
        self.discovered_input_plugins = {
            entry_point.name: entry_point.load()
            for entry_point
            in pkg_resources.iter_entry_points('feature_eng.plugins_input')
        }
        self.discovered_output_plugins = {
            entry_point.name: entry_point.load()
            for entry_point
            in pkg_resources.iter_entry_points('feature_eng.plugins_output')
        }
        self.discovered_core_plugins = {
            entry_point.name: entry_point.load()
            for entry_point
            in pkg_resources.iter_entry_points('feature_eng.plugins_core')
        }

    def load_plugins(self):
        """ Loads plugin entry points into class attributes"""
        if self.conf.input_plugin in self.discovered_input_plugins:
            self.ep_i = self.discovered_input_plugins[self.conf.input_plugin]
            if self.conf.args == None:
                # TODO: QUITAR
                _logger.debug("initializing input plugin via constructor.")
            else:
                # if using command line (conf == None), uses unknown parameters from arparser as params for plugins
                _logger.debug("initializing input plugin via command line parameters.")
            self.ep_input = self.ep_i(self.conf)
        else:
            print("Error: Input Plugin not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()
        if self.conf.output_plugin in self.discovered_output_plugins:
            self.ep_o = self.discovered_output_plugins[self.conf.output_plugin]
            self.ep_output = self.ep_o(self.conf)
        else:
            print("Error: Output Plugin not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()
        if self.conf.core_plugin in self.discovered_core_plugins:
            self.ep_c = self.discovered_core_plugins[self.conf.core_plugin]
            self.ep_core = self.ep_c(self.conf)
        else:
            print("Error: Core Plugin not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()
    
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
        
def run(args):
    """ Entry point for console_scripts """
    feature_eng = FeatureEng(None)
    feature_eng.main(args)

if __name__ == "__main__":
    run(sys.argv)
