# Feature Engineering

Transform raw data to generate new data that better represent features so they improve the performance of a predictive model. __Work In Progress, NOT USABLE YET__.

[![Build Status](https://travis-ci.org/harveybc/feature-engineering.svg?branch=master)](https://travis-ci.org/harveybc/feature-engineering)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-feature-engineering.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/feature-engineering?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/feature-engineering/blob/master/LICENSE)

## Description

Implements modular components for featuer engineering: a heuristic training signal generator, a MSSA decomposer and a MSSA predictor. 

All modules are usable both from command line and from class methods.

## Installation

The installation is made by clonning the github repo and manually installing it, package based installation coming soon..

### Steps
1. Clone the GithHub repo:   
> git clone https://github.com/harveybc/feature-engineering
2. Change to the repo folder:
> cd feature-engineering
3. Install requirements.
> pip install -r requirements.txt
4. Install python package (also installs the console command data-trimmer)
> python setup.py install
5. (Optional) Perform tests
> python setup.py install
6. (Optional) Generate Sphinx Documentation
> python setup.py docs

## Modules

All the class modules and their CLI commands are installed with the feature-engineering package, the following sections describe each module briefly and link to each module's basic documentation. 

Additional detailed Sphinix documentation for all modules can be generated in HTML format with the optional step 6 of the installation process, it contains documentation of the classes and methods of all modules in the feature-engineering package. 

## Heuristic Training Signal Generator

Generates an ideal training signal for trading using EMA_fast forwarded a number of ticks minus current EMA_slow as buy signal.

See [heuristic_ts Readme](../master/README_heuristic_ts.md) for detailed description and usage instructions.

## Multivariate Singular Spectrum Analysis (MSSA) Decomposer. 

Performs MSSA decomposition, save the output dataset containing a configurable number of components per feature or the sum of a configurable number of components.

See [MSSA Decomposer Readme](../master/README_mssa_decomposer.md) for detailed description and usage instructions.

## MSSA Predictor

Performs MSSA prediction for a configurable number of forward ticks, save the .output dataset containing the prediction for a configurable number of channels or its sum.

See [MSSA Predictor Readme](../master/README_mssa_predictor.md) for detailed description and usage instructions.

## Examples of usage

The following examples show both the class method and command line uses for one module, for examples of other modules, please see the specific module´s documentation.

### Example: Usage via Class Methods (HeuristicTS)
```python
from feature-engineering.heuristic_ts.heuristic_ts import HeuristicTS
# configure parameters (same variable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
conf = Conf()
# instance class and loads dataset
dt = HeuristicTS(conf)
# execute the module´s core method
dt.core()
# save output to output file (defaults to input file with .output extension)
dt.store()
```

### Example: Usage via CLI (DataTrimmer)

> heuristic_ts --input_file "tests/data/test_input.csv"






