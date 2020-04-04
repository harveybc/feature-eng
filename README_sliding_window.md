# Preprocessor: Sliding Window

A simple data pre-processor that performs sliding window on a timeseries dataset. Usable both from command line and from class methods.

[![Build Status](https://travis-ci.org/harveybc/preprocessor.svg?branch=master)](https://travis-ci.org/harveybc/preprocessor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-preprocessor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/preprocessor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/preprocessor/blob/master/LICENSE)

## Description

Performs timeseries windowing for an input dataset. Uses a configurable window size and generates a window of past data per tick
of the input dataset.

Usable both from command line and from class methods.

The sliding_window is implemented in the SlidingWindow class, it has methods for loading a dataset, performing sliding window via the windowize ()  method for producing an output dataset, please see [test_sliding_window](https://github.com/harveybc/preprocessor/blob/master/tests/sliding_window/test_sliding_window.py). 

## Installation

The module is installed with the preprocessor package, the instructions are described in the following section.

### Steps
1. Clone the GithHub repo:   
> git clone https://github.com/harveybc/preprocessor
2. Change to the repo folder:
> cd preprocessor
3. Install requirements.
> pip install -r requirements.txt
4. Install python package (also installs the console command data-trimmer)
> python setup.py install
5. (Optional) Perform tests
> python setup.py install
6. (Optional) Generate Sphinx Documentation
> python setup.py docs

### Command-Line Execution

The sliding window is also implemented as a console command:
> sliding_window -- input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset to be processed.
* __--output_file <filename>__: (Optional) Filename for the output dataset. Defaults to the input dataset with the .output extension.
* __--window_size <filename>__: (Optional) Size of the sliding window, defaults to 21.

## Examples of usage
The following examples show both the class method and command line uses.

### Usage via Class Methods
```python
from preprocessor.sliding_window.sliding_window import SlidingWindow
# configure parameters (same vaiable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
conf = Conf()
# instance trimmer class and loads dataset
st = SlidingWindow(conf)
# do the trimming
st.windowize()
# save output to output file
st.store()
```

### Usage via CLI

> sliding_window --input_file "tests/data/test_input.csv"






