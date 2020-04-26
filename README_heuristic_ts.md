# Feature Engineering: heuristic_ts

Generates an ideal training signal for trading based on a feature forwarded a configurable number of ticks.

[![Build Status](https://travis-ci.org/harveybc/preprocessor.svg?branch=master)](https://travis-ci.org/harveybc/preprocessor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-preprocessor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/preprocessor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/preprocessor/blob/master/LICENSE)

## Description

Generates a training signal using Exponentioal Moving Average (EMA) fast, forwarded a configurable number of ticks minus EMA slow.  The input dataset must contain the two EMA columns.  The column index of the EMA fast and slow are configurable.

## Installation

The module is installed with the preprocessor package, the instructions are described in the [preprocessor README](../master/README.md).

### Command-Line Execution

The heuristic_ts also is implemented as a console command:
> heuristic_ts -- input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset to be trimmed.
* __--output_file <filename>__: (Optional) Filename for the output dataset. Defaults to the input dataset with the .output extension.
* __--ema_fast <val>__:(Optional) column index of the EMA fast in the input dataset. Defaults to 0.
* __--ema_slow <val>__: (Optional) column index of the EMA slow in the input dataset. Defaults to 1.
* __--forward_ticks <val>__: (Optional) Number of forward ticks for EMA fast.
* __--group_similar__: (Optional) Group by summing the similar channels.

## Examples of usage
The following examples show both the class method and command line uses.

### Usage via Class Methods
```python
from preprocessor.heuristic_ts.heuristic_ts import HeurusticTS
# configure parameters (same variable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
conf = Conf()
# instance class and loads dataset
dt = HeurusticTS(conf)
# process the data
dt.core()
# save output to output file
dt.store()
```

### Usage via CLI

> heuristic_ts --input_file "tests/data/test_input.csv"






.