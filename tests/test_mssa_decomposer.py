# -*- coding: utf-8 -*-

import pytest
import csv
import sys
import os
from filecmp import cmp
from feature_eng.feature_eng import FeatureEng

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

class Conf:
    """ This method initialize the configuration variables for a plugin """
    
    def __init__(self):
        """ Component Tests Constructor """
        self.input_file = os.path.join(os.path.dirname(__file__), "data/test_input.csv")
        """ Test dataset filename """
        self.output_file = os.path.join(os.path.dirname(__file__), "data/test_output.csv")
        """ Output dataset filename """
        self.list_plugins = False
        self.core_plugin = "mssa_decomposer"
        self.num_components = 5
        self.window_size = 30
        self.group_file = None
        self.plot_prefix = None
        self.w_prefix = None

class TestMSSADecomposer:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.rows_d, self.cols_d = self.get_size_csv(self.conf.input_file)
        """ Get the number of rows and columns of the test dataset """
        try:
            os.remove(self.conf.output_file)
        except:
            print("No test output file found.")
            pass

    def get_size_csv(self, csv_file):
        """ Get the number of rows and columns of a test dataset, used in all tests.

        Args:
        csv_file (string): Path and filename of a test dataset

        Returns:
        (int,int): number of rows, number of columns
        """
        rows = list(csv.reader(open(csv_file)))
        return (len(rows), len(rows[0]))


    def test_C03T01_core(self):
        """ Loads plugin from FeatureEng using parameters from setup_method() and Asses that output file has 1 column and num_ticks - forward_ticks """
        self.fe = FeatureEng(self.conf)
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assertion
        assert (cols_o == self.fe.ep_core.cols_d * self.conf.num_components)

    def test_C03T02_cmdline(self):
        """ same as C03T02, but via command-line """
        os.system("feature_eng --core_plugin mssa_decomposer --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --num_components "
            + str(self.conf.num_components)
        )
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert (cols_o == self.cols_d * self.conf.num_components)

    def test_C03T03_group_file(self):
        """ assert if there are 3 groups per feature in the output dataset """
        os.system("feature_eng --core_plugin mssa_decomposer --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --num_components "
            + str(self.conf.num_components)
            + " --group_file "
            + os.path.join(os.path.dirname(__file__), "data/groups.json")
        ) 
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if there are 3 groups per feature in the output dataset
        assert (cols_o == self.cols_d * 3)

    def test_C03T04_w_prefix(self):
        """ assert if there are 3 groups per feature in the output dataset """
        os.system("feature_eng --core_plugin mssa_decomposer --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --num_components "
            + str(self.conf.num_components)
            + " --w_prefix "
            + os.path.join(os.path.dirname(__file__), "data/w_")
        ) 
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if there are 3 groups per feature in the output dataset
        assert (cols_o == self.cols_d * self.conf.num_component)
    
    def test_C03T05_w_prefix_group_file(self):
        """ assert if there are 3 groups per feature in the output dataset """
        os.system("feature_eng --core_plugin mssa_decomposer --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --num_components "
            + str(self.conf.num_components)
            + " --w_prefix "
            + os.path.join(os.path.dirname(__file__), "data/w_")
            + " --group_file "
            + os.path.join(os.path.dirname(__file__), "data/groups.json")
        ) 
        # get the size of the output dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if there are 3 groups per feature in the output dataset
        assert (cols_o == self.cols_d * self.conf.num_component)