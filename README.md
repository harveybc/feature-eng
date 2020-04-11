# Feature Engineering

Transform raw data to generate new data that better represent features so they improve the performance of a predictive model. Usable both from command line and from class methods. __Work In Progress, NOT USABLE YET__.

[![Build Status](https://travis-ci.org/harveybc/feature-engineering.svg?branch=master)](https://travis-ci.org/harveybc/feature-engineering)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-feature-engineering.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/feature-engineering?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/feature-engineering/blob/master/LICENSE)

## Description

Implements modular components for dataset preprocessing: a data-trimmer, a standardizer, a feature selector and a sliding window data generator.

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

All the CLI commands and the class modules are installed with the feature-engineering package, the following sections describe each module briefly and link to each module's basic documentation. 

Detailed Sphinix documentation for all modules can be generated in HTML format with the optional step 6 of the installation process, it contains documentation of the classes and methods of all modules in the feature-engineering package. 

## Data-Trimmer

A simple data pre-processor that trims the constant valued columns.  Also removes rows from the start and the end of a dataset with features with consecutive zeroes. Usable both from command line and from class methods.

See [Data-Trimmer Readme](../master/README_data_trimmer.md) for detailed description and usage instructions.

## Standarizer

Standardizes a dataset and exports the standarization configuration for use on other datasets. Usable both from command line and from class methods.

See [Standardizer Readme](../master/README_standardizer.md) for detailed description and usage instructions.

## Sliding Window

Performs the sliding window technique. Usable both from command line and from class methods.

See [Sliding Window Readme](../master/README_sliding_window.md) for detailed description and usage instructions.

## Feature Selector

Performs the feature selection based on a classification or regression training signal and a threeshold. Usable both from command line and from class methods.

See [Feature Selector Readme](../master/README_feature_selector.md) for detailed description and usage instructions.

## Examples of usage

The following examples show both the class method and command line uses for the data-trimmer feature-engineering module, please see the documentation for examples of other modules.

### Example: Usage via Class Methods (DataTrimmer)
```python
from feature-engineering.data_trimmer.data_trimmer import DataTrimmer
# configure parameters (same vaiable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
conf = Conf()
# instance trimmer class and loads dataset
dt = DataTrimmer(conf)
# do the trimming
rows_t, cols_t = dt.trim_auto()
# save output to output file
dt.store()
```

### Example: Usage via CLI (DataTrimmer)

> data-trimmer --input_file "tests/data/test_input.csv"






