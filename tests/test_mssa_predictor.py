# -*- coding: utf-8 -*-

import pytest
import csv
import sys
import os
import filecmp
import numpy as np

from feature_engineering.sliding_window.sliding_window import SlidingWindow

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
        self.window_size = 21
        """ Output configuration of the feature_engineering """

class TestSlidingWindow:
    """ Component Tests  """

    def setup_method(self, test_method):
        """ Component Tests Constructor """
        self.conf = Conf()
        self.dt = SlidingWindow(self.conf)
        """ Data sliding_window object """
        try:
            os.remove(self.conf.output_file)
        except:
            print("No test output file found.")
            pass

    def atest_C03T01_window(self):
        """ Perform sliding_window and assert if the output_columns == input_columns * (window_size-1) """        
        self.dt.window()
        # save output to file
        self.dt.store()
        #TODO: PROCESS BEFORE ASSERT
        assert output_columns == (input_columns * (self.conf.window_size-1))

    def atest_C03T02_cmdline_window(self):
        """ Perform the same C03T01_window assertion but using command line arguments """
        os.system(
            "sliding_window --input_file "
            + self.conf.input_file
            + " --output_file "
            + self.conf.output_file
        )
        #TODO: PROCESS BEFORE ASSERT
        assert output_columns == (input_columns * (self.conf.window_size-1))
