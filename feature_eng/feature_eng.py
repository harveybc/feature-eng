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
        self.setup_logging(logging.DEBUG)
        _logger.info("Starting feature_eng...")
        if conf != None:
            self.input_file = conf.input_file
            """ Path of the input dataset """
            if hasattr(conf, "output_file"):
                self.output_file = conf.output_file
            else:
                self.output_file = self.input_file + ".output"
            """ Path of the output dataset """
            if hasattr(conf, "input_config_file"):
                self.input_config_file = conf.input_config_file
            else:
                self.input_config_file = None
            """ Path of the input configuration """
            if hasattr(conf, "output_config_file"):
                self.output_config_file = conf.output_config_file
            else:
                self.output_config_file = None
            """ Path of the output configuration """
            if hasattr(conf, "plugin"):
                self.plugin = conf.plugin
                # Load plugin
                _logger.debug("Finding Plugins.")
                self.find_plugins()
                # Load plugin
                _logger.debug("Loading plugin.")
                self.load_plugin()
                self.fep = self.plugin_entry_point:FeatureEngPlugin(conf)
                # Load input dataset
                _logger.debug("Loading input file.")
            else:
                self.plugin = None
            """ Plugin to load """
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
        self.parse_args(args)
        if self.plugin != None:    
            # Load plugin
            _logger.debug("Finding Plugins.")
            self.find_plugins()
            # Load plugin
            _logger.debug("Loading plugin.")
            self.load_plugin()
            # Instantiate plugin class
            self.fep = self.plugin_entry_point.FeatureEngPlugin()
            # Load input dataset
            if self.input_ds == None:
                _logger.debug("Loading input file.")
                #self.load_ds()
                self.fep.load_ds()
            # Start core function
            #self.core()
            self.fep.core()
            # Start logger
            _logger.debug("Saving results.")
            # Save results and output configuration
            self.fep.store()
        else:
            if self.list_plugins == True:
                self.find_plugins()
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
        for key, value in self.discovered_plugins:
            print(key+"\n")

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

def run(args):
    """ Entry point for console_scripts """
    feature_eng = FeatureEng(None)
    feature_eng.main(args)


if __name__ == "__main__":
    run(sys.argv)
