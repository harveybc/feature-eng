# FeatureEng: MSSA-Decomposer

Performs Multivariate Singular Spectrum Analysis (MSSA) decomposition of an input dataset.

[![Build Status](https://travis-ci.org/harveybc/feature_eng.svg?branch=master)](https://travis-ci.org/harveybc/feature_eng)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-feature_eng.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/feature_eng?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/feature_eng/blob/master/LICENSE)

## Description

Performs MSSA decomposition of an input dataset, uses a configurable number of output channels and optionally 
grouping similar components.

The mssa_decomposer is implemented in the MSSADecomposer class, it has methods for loading a dataset processing it an producing an output, please see [test_mssa_decomposer](https://github.com/harveybc/feature_eng/blob/master/tests/mssa_decomposer/test_mssa_decomposer.py)

It also saves a configuration file, that is a CSV files with removed files and columns for applying similar  trimming to another dataset. Usable both from command line and from class methods (see [tests folder](https://github.com/harveybc/feature_eng/tree/master/tests)).

## Installation

The module is installed with the feature_eng package, the instructions are described in the [feature_eng README](../master/README.md).

### Command-Line Execution

The mssa_decomposer also is implemented as a console command:
> mssa_decomposer -- input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset to be trimmed.
* __--output_file <filename>__: (Optional) Filename for the output dataset. Defaults to the input dataset with the .output extension.
* __--output_config_file <filename>__: (Optional) Filename for the output configuration containing rows trimmed in columns 0 and columns trimmed in column 1. Defaults to the input dataset with the .config extension.
* __--input_config_file <filename>__: (Optional) Imports an existing configuration and trims a dataset with it.
* __--from_start <val>__:(Optional) number of rows to remove from the start of the input dataset.
* __--from_end <val>__: (Optional) number of rows to remove from the end of the input dataset.
* __--remove_columns__: (Optional) Removes all constant columns.
* __--no_auto_trim__: (Optional) Do not perform auto-trimming, useful if using the remove_columns, from_start or from_end options.

## Examples of usage
The following examples show both the class method and command line uses.

### Usage via Class Methods
```python
from feature_eng.mssa_decomposer.mssa_decomposer import MSSADecomposer
# configure parameters (same vaiable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
conf = Conf()
# instance trimmer class and loads dataset
dt = MSSADecomposer(conf)
# do the trimming
rows_t, cols_t = dt.trim_auto()
# save output to output file
dt.store()
```

### Usage via CLI

> mssa_decomposer --input_file "tests/data/test_input.csv"






