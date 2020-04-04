# -*- coding: utf-8 -*-
"""
This File contains the DataTrimmer class. To run this script uncomment or add the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
        data-trimmer = data_trimmer.__main__:main

Then run `python setup.py install` which will install the command `data-trimmer`
inside your current environment.

"""

import argparse
import sys
import logging
import numpy as np
from preprocessor.preprocessor import Preprocessor
from itertools import zip_longest 

# from data_trimmer import __version__

__author__ = "Harvey Bastidas"
__copyright__ = "Harvey Bastidas"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class DataTrimmer(Preprocessor):
    """ The Data Trimmer preprocessor class """

    def __init__(self, conf):
        """ Constructor using same parameters as base class """
        super().__init__(conf)

    def parse_args(self, args):
        """ Parse command line parameters

        Args:
            args ([str]): command line parameters as list of strings

        Returns:
            :obj:`argparse.Namespace`: command line parameters namespace
        """
        parser = argparse.ArgumentParser(
            description="Dataset Trimmer: trims constant columns and consecutive zero rows from the end and the start of a dataset."
        )
        parser.add_argument(
            "--from_start",
            help="number of rows to remove from start (ignored if auto_trim)",
            type=int,
            default=0
        )
        parser.add_argument("--from_end",
            help="number of rows to remove from end (ignored if auto_trim)",
            type=int,
            default=0
        )
        parser.add_argument("--remove_columns", 
            help="removes constant columns", 
            action="store_true",
            default=False
        )
        parser.add_argument("--no_auto_trim",
            help="trims the constant columns and trims all rows with consecutive zeroes from start and end",
            action="store_true",
            default=False
        )
        parser = self.parse_cmd(parser)
        pargs = parser.parse_args(args)
        self.assign_arguments(pargs)
        if hasattr(pargs, "from_start"):
            self.from_start = pargs.from_start
        if hasattr(pargs, "from_end"):
            self.from_end = pargs.from_end
        if hasattr(pargs, "remove_columns"):
            self.remove_columns = pargs.remove_columns
        if hasattr(pargs, "no_auto_trim"):
            self.auto_trim = not(pargs.no_auto_trim)
        else:
            self.auto_trim = True

    def core(self):
        """ Core preprocessor task after starting the instance with the main method.
            Decide from the arguments, what trimming method to call.

        Args:
        args (obj): command line parameters as objects
        """
        if (self.from_start >= 0) and (self.from_end >= 0):
            self.trim_fixed_rows(self.from_start, self.from_end)
        if self.remove_columns:
            self.trim_columns()
        if self.auto_trim:
            self.trim_auto()
        if hasattr(self, "input_config_file"):
            if self.input_config_file != None:
                self.config_ds = np.genfromtxt(self.input_config_file, delimiter=",")
                self.load_from_config()
        
    def trim_fixed_rows(self, from_start, from_end):
        """ Trims a configurable number of rows from the start or end of the input dataset

        Args:
            from_start (int): number of rows to remove from start (ignored if auto_trim)
            from_end (int): number of rows to remove from end (ignored if auto_trim)

        Returns:
            rows_t, cols_t (int,int): number of rows and columns trimmed
        """
        # remove from start
        self.output_ds = self.input_ds[from_start : len(self.input_ds), :]
        self.r_rows = list(range(0, from_start))
        # remove from end
        self.output_ds = self.output_ds[: len(self.output_ds) - from_end, :]
        self.r_rows = self.r_rows + list(range(self.rows_d - from_end, self.rows_d))
        # assign output as new input for performing consecutive trimming of columns
        if hasattr(self, "remove_columns"):
            if self.remove_columns:
                self.input_ds = np.copy(self.output_ds)
        return from_end + from_start, 0

    def trim_columns(self):
        """ Trims all the constant columns from the input dataset

        Returns:
            rows_t, cols_t (int,int): number of rows and columns trimmed
        """
        self.rows_d, self.cols_d = self.input_ds.shape
        # initialize unchanged_array as true with size num_columns
        un_array = np.array([True] * self.cols_d)
        # in two consecutive rows, search the unchanged values
        for i in range(self.rows_d - 1):
            unchanged = np.equal(self.input_ds[i, :], self.input_ds[i + 1, :])
            # for each un_array that is true, if the values changed, set it to false
            un_array = np.logical_and(un_array, unchanged)
        # remove all rows with true on the un_array
        self.output_ds = self.input_ds[:, np.logical_not(un_array)]
        # generate an array with the indexes of the rows marked with true in un_array
        cols = np.nonzero(un_array)
        self.r_cols = cols[0]
        # assign output as new input for performing consecutive auto trimming 
        if hasattr(self, "auto_trim"):
            if self.auto_trim:
                self.input_ds = np.copy(self.output_ds)
        return 0, np.sum(un_array)

    def trim_auto(self):
        """ Trims all the constant columns and trims all rows with consecutive zeroes from start and end of the input dataset

        Returns:
        rows_t, cols_t (int,int): number of rows and columns trimmed
        """
        
        self.rows_d, self.cols_d = self.input_ds.shape
        rows_t, cols_t = self.trim_columns()
        # delete rows from start that contain zeroes from start
        z_array = self.output_ds[0] == 0
        c_add = 0
        while np.any(z_array):
            c_add = c_add + 1
            rows_t = rows_t + 1
            # delete the first row of the output_ds and updates z_array
            self.output_ds = np.delete(self.output_ds, [0], axis=0)
            z_array = self.output_ds[0] == 0
        self.r_rows = self.r_rows + list(range(0,c_add))
        return rows_t, cols_t

    def load_from_config(self):
        # get the number of rows in the config_ds
        n_rows = len(self.config_ds)
        # update rrows and rcols
        self.r_rows=self.config_ds[:, 0]
        self.r_cols=self.config_ds[:, 1]
        # replace -1 in the config_ds with None
        self.cr_rows = [None if int(x)==-1 else int(x) for x in [self.config_ds[:, 0]]]
        self.cr_cols = [None if int(x)==-1 else int(x) for x in [self.config_ds[:, 1]]]
        # convert each column to binary array
        self.br_rows = np.zeros(n_rows) 
        self.br_cols = np.zeros(n_rows) 
        self.br_rows[self.cr_rows] = 1
        self.br_cols[self.cr_cols] = 1
        # remove the rows marked with true from the input_ds in the first array from the config file
        self.output_ds = self.input_ds[np.logical_not(self.r_rows), :] 
        # remove the columns marked with true in the second array from the config file
        self.output_ds = self.output_ds[:, np.logical_not(self.r_cols)] 
        
        
    def store(self):
        """ Save preprocessed data and the configuration of the preprocessor. """
        print("self.output_ds.shape = ", self.output_ds.shape)
        config_rows = list(zip_longest(self.r_rows, self.r_cols, fillvalue=-1))
        _logger.debug("output_file = "+ self.output_file)
        np.savetxt(self.output_file, self.output_ds, delimiter=",")
        if (self.output_config_file == None):
            self.output_config_file = self.input_file + ".config"
        _logger.debug("ocf = "+ self.output_config_file)
        np.savetxt(self.output_config_file, config_rows, delimiter=",")
        

def run(args):
    """ Entry point for console_scripts """
    data_trimmer = DataTrimmer(None)
    data_trimmer.main(args)


if __name__ == "__main__":
    run(sys.argv)
