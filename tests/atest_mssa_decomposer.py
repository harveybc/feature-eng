# -*- coding: utf-8 -*-


import pytest
import csv
import sys
import os
import filecmp
import numpy as np

from feature_eng.standardizer.standardizer import Standardizer

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"


class Conf:
    def __init__(self):
        """ Component Tests Constructor """
        fname = os.path.join(os.path.dirname(__file__), "data/test_input.csv")
        self.input_file = fname
        """ Test dataset filename """
        fname = os.path.join(os.path.dirname(__file__), "data/test_output.csv")
        self.output_file = fname
        """ Output dataset filename """
        fname = os.path.join(os.path.dirname(__file__), "data/out_config.csv")
        self.output_config_file = fname
        """ Output configuration of the feature_eng """

class TestStandardizer:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.dt = Standardizer(self.conf)
        """ Data standardizer object """
        try:
            os.remove(self.conf.output_file)
        except:
            print("No test output file found.")
            pass

    def test_C02T01_standardize(self):
        """ Standardizes all the columns and assert if the average is near zero """        
        self.dt.standardize()
        # save output to file
        self.dt.store()
        mean_all = np.mean(self.dt.output_ds, dtype=np.float64)
        assert (mean_all>-10) and (mean_all<10)

    def test_C02T02_cmdline_standarize(self):
        """ Standardizes all the columns using command line arguments """
        os.system(
            "standardizer --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
        )
        # read the output file
        o_array = np.genfromtxt(self.conf.output_file, delimiter=",")
        mean_all = np.mean(o_array[:,0], dtype=np.float64)
        assert (mean_all>-10) and (mean_all<10)

    def test_C02T03_config_save_load(self):
        """ Save a configuration file and uses it to standardize a dataset. Assert that output_config can be loaded and the output_config(loaded) == output_config(saved)"""
        os.system(
            "standardizer --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --output_config_file "
            + self.conf.output_config_file 
        )
        # Uses the output as input for another dataset and compare with desired output.
        os.system(
            "standardizer --input_file "
            + self.conf.input_file
            + " --input_config_file "
            + self.conf.output_config_file
            + " --output_file "
            + self.conf.output_file + ".c03t03"
            + " --output_config_file "
            + self.conf.output_config_file 
        )
        assert filecmp.cmp(self.conf.output_file, self.conf.output_file  + ".c03t03", shallow=True)