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
        