# -*- coding: utf-8 -*-
# Palomita, i miss you, Rest in Peace my love, soon i'll get to you and give you a kiss in your forehead :(

import pytest
import csv
import sys
import os
from filecmp import cmp

from feature_eng.heuristic_ts.heuristic_ts import HeuristicTS

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

class TestHeuristicTS:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.dt = HeuristicTS(self.conf)
        """ Data trimmer object """
        self.rows_d, self.cols_d = self.get_size_csv(self.conf.input_file)
        """ Get the number of rows and columns of the test dataset """
        self.forward_ticks = 5
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

    def test_C05T01_training_signal(self):
        """ Asses that output file has 1 column and num_ticks - forward_ticks """
        rows_t, cols_t = self.dt.core()
        # save output to file
        self.dt.store()
        # get the number of rows and cols from out_file
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the new == old - trimmed
        assert (cols_o == 1) and (rows_o = rows_t - self.forward_ticks)

    def test_C05T02_cmdline_training_signal(self):
        """ same as C03T02, but via command-line """
        os.system("euristic_ts --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
        )
        # get the size of the original dataset
        rows_d, cols_d = self.get_size_csv(self.conf.input_file)
        # get the size of the output dataset
        rows_o, cols_o = self.get_size_csv(self.conf.output_file)
        # assert if the number of rows an colums is less than the input dataset and > 0
        assert (cols_o == 1) and (rows_o = rows_t - self.forward_ticks)
        