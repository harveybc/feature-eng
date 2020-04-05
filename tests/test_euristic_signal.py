# -*- coding: utf-8 -*-


import pytest
import csv
import sys
import os
from filecmp import cmp

from feature_engineering.data_trimmer.data_trimmer import DataTrimmer

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
        # fname = os.path.join(os.path.dirname(__file__), "../data/in_config.csv")
        # self.input_config_file = fname
        #""" Input configuration of the proprocessor """
        fname = os.path.join(os.path.dirname(__file__), "data/out_config.csv")
        self.output_config_file = fname
        """ Output configuration of the proprocessor """


class TestDataTrimmer:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.dt = DataTrimmer(self.conf)
        """ Data trimmer object """
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

    def test_C01T01_trim_fixed_rows(self):
        """ Trims a configurable number of rows from the start or end of the input dataset by using the trim_fixed_rows method. Execute trimmer with from_start=10, from_end=10. """
        rows_t, cols_t = self.dt.trim_fixed_rows(10, 10)
        # save output to file
        self.dt.store()
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (self.rows_d + self.cols_d) - (rows_t + cols_t)

    def test_C01T02_trim_columns(self):
        """ Trims all the constant columns by using the trim_columns method. Execute trimmer with remove_colums = true. """
        rows_t, cols_t = self.dt.trim_columns()
        # save output to file
        self.dt.store()
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (self.rows_d + self.cols_d) - (rows_t + cols_t)

    def test_C01T03_trim_auto(self):
        """ Trims all the constant columns and trims all rows with consecutive zeroes from start and end by using the trim_auto method. Execute trimmer with auto_trim = true.  """
        rows_t, cols_t = self.dt.trim_auto()
        # save output to file
        self.dt.store()
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the new == old - trimmed
        assert (rows_o + cols_o) == (self.rows_d + self.cols_d) - (rows_t + cols_t)

    def test_C01T04_cmdline_remove_columns(self):
        """ Trims all the constant columns using command line arguments """
        os.system(
            "data-trimmer --remove_columns --no_auto_trim --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
        )
        # get the size of the original dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert ((cols_d - cols_o) > 0) and ((cols_o > 0) and (rows_o > 0))

    def test_C01T05_cmdline_remove_columns_rows(self):
        """ Trims all the constant columns and 10  rows from start and end using command line arguments """
        os.system(
            "data-trimmer --from_start 10 --from_end 10 --remove_columns --no_auto_trim --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
        )
        # get the size of the original dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert ((cols_d - cols_o) > 0) and ((rows_d - rows_o) > 0) and ((cols_o > 0) and (rows_o > 0))

    def test_C01T06_cmdline_remove_columns_rows_auto(self):
        """ Trims all the constant columns and 0 rows from start and end (for pipeline) and auto trimming  using command line arguments """
        os.system(
            "data-trimmer --from_start 0 --from_end 0 --remove_columns --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
        )
        # get the size of the original dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert ((cols_d - cols_o) > 0) and ((rows_d - rows_o) > 0) and ((cols_o > 0) and (rows_o > 0))

    def test_C01T07_config_save(self):
        """ Save a configuration file and uses it to trim a dataset. Assert that output_config can be loaded and the output_config(loaded) == output_config(saved)"""
        os.system(
            "data-trimmer --input_file --from_start 20"
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
            + " --output_config_file "
            + self.conf.output_config_file 
        )
        # Uses the output as input for another dataset and compare with desired output.
        os.system(
            "data-trimmer --input_file "
            + self.conf.input_file
            + " --input_config_file "
            + self.conf.output_config_file
            + " --output_file "
            + self.conf.output_file
            + " --output_config_file "
            + self.conf.output_config_file  + ".c02t07"
        )
        assert cmp(self.conf.output_config_file, self.conf.output_config_file  + ".c02t07", shallow=True)

    def test_C01T08_config_load(self):
        """ Load a configuration file and uses it to trim a dataset. Verify that output_config == input_config"""
        fname = os.path.join(os.path.dirname(__file__), "data/in_config.csv")
        input_config_file = fname
        """ Input configuration of the proprocessor """
        os.system(
            "data-trimmer --input_file "
            + self.conf.input_file
            + " --input_config_file "
            +  input_config_file
            + " --output_file "
            + self.conf.output_file
            + " --output_config_file "
            + self.conf.output_config_file
        )
        assert cmp(input_config_file, self.conf.output_config_file, shallow=True)
