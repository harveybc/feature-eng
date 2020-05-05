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
            if hasattr(conf, "input_plugin"):
                self.input_plugin = conf.input_plugin
            else:
                self.input_plugin = "csv_input"
            """ Name of the input plugin """
            if hasattr(conf, "output_plugin"):
                self.output_plugin = conf.output_plugin
            else:
                self.output_plugin = "csv_output"
            """ Name of the output plugin """
            if hasattr(conf, "core_plugin"):
                self.core(conf)
            else:
                self.core_plugin = None
            if hasattr(conf, "list_plugins"):
                self.list_plugins = True
                _logger.debug("Listing plugins.")
                self.find_plugins()
                _logger.debug("Priniting plugins.")
                self.print_plugins()
            else:
                self.list_plugins = False
            """ If true, lists all installed external and internal plugins. """
        else :
            self.input_ds = None
        self.r_rows = []
        self.r_cols = []
        self.config_ds = None

    def parse_cmd(self, parser):
        parser.add_argument("--version", action="version", version="feature_eng")
        parser.add_argument("--list_plugins", help="lists all installed external and internal plugins")
        parser.add_argument("--plugin", help="Plugin to load ")
        parser.add_argument("--input_file", help="Input CSV filename ")
        parser.add_argument("--output_file", help="Output CSV filename")
        parser.add_argument("--input_config_file", help="Input configuration  filename")
        parser.add_argument("--output_config_file", help="Output configuration  filename")
        parser.add_argument("-v","--verbose",dest="loglevel",help="set loglevel to INFO",action="store_const",const=logging.INFO)
        parser.add_argument("-vv","--very_verbose",dest="loglevel",help="set loglevel to DEBUG",action="store_const",const=logging.DEBUG)
        return parser
    
    def assign_arguments(self,pargs):
        self.list_plugins =  False
        if hasattr(pargs, "plugin"):
            self.plugin = pargs.plugin
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
        elif hasattr(pargs, "list_plugins"):
            self.list_plugins = pargs.list_plugins
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
        conf = self.parse_args(args)
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

    def load_plugin(self):
        if self.plugin in self.discovered_plugins:
            self.plugin_entry_point = self.discovered_plugins[self.plugin]
        else:
            print("Error: Plugin "+ self.plugin +" not found. Use option --list_plugins to show the list of available plugins.")
            sys.exit()

    def find_plugins(self):
        self.discovered_plugins = {
            entry_point.name: entry_point.load()
            for entry_point
            in pkg_resources.iter_entry_points('feature_eng.plugins')
        }

    def print_plugins(self):
        for key in self.discovered_plugins:
            print(key+"\n")

    def store(self):
        """ Save preprocessed data and the configuration of the feature_eng. """
        pass

    def core(self,conf):
        """ Core feature_eng task after starting the instance with the main method.
            To be overriden by child classes depending on their feature_eng task.
        """
        self.core_plugin = conf.core_plugin
        """ Core plugin to load """
        _logger.debug("Finding Plugins.")
        self.find_plugins()
        _logger.debug("Loading plugins.")
        self.load_plugins()
        _logger.debug("Initializing plugins.")
        self.init_plugins(conf)
        _logger.debug("Loading input dataset from the input plugin.")
        self.input_ds = self.ep_input.load_ds() 
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
        pargs, unknown = parser.parse_known_args(args)
        self.assign_arguments(pargs)
        
def run(args):
    """ Entry point for console_scripts """
    feature_eng = FeatureEng(None)
    feature_eng.main(args)


if __name__ == "__main__":
    run(sys.argv)
