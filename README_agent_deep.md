# Agent: AgentDeep

Agent for an OpenAI Gym environment, implementing a Keras model as feature extractor.

[![Build Status](https://travis-ci.org/harveybc/preprocessor.svg?branch=master)](https://travis-ci.org/harveybc/preprocessor)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-preprocessor.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/preprocessor?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/preprocessor/blob/master/LICENSE)

## Description

This agent loads an standardized observation dataset on a gym environment, 
reads observations from it and executes actions based on the output 
of a Keras model used as feature extractor, and a neat-python pre-optimized neural network 
used as action controller.

Exports a csv containing the agent state (balance, equity, orders status) and the trading history. 

The agent is implemented in the AgentDeep class, it has methods for loading a dataset on a simulation environment, performing steps of the simulation and d it and producing an output dataset and a configuration file that can be loaded and applied to another dataset, please see [test_agent_deep](https://github.com/harveybc/agent/blob/master/tests/standardizer/test_standardizer.py). It can also be used via command line.

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

The standardizer also is implemented as a console command:
> standardizer -- input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset to be trimmed.
* __--output_file <filename>__: (Optional) Filename for the output dataset. Defaults to the input dataset with the .output extension.
* __--output_config_file <filename>__: (Optional) Filename for the output configuration containing rows trimmed in columns 0 and columns trimmed in column 1. Defaults to the input dataset with the .config extension.
* __--input_config_file <filename>__: (Optional) Imports an existing configuration and trims a dataset with it.

## Examples of usage
The following examples show both the class method and command line uses.

### Usage via Class Methods
```python
from preprocessor.standardizer.standardizer import Standardizer
# configure parameters (same vaiable names as command-line parameters)
class Conf:
    def __init__(self):
        self.input_file = "tests/data/test_input.csv"
conf = Conf()
# instance trimmer class and loads dataset
st = Standardizer(conf)
# do the trimming
st.standardize()
# save output to output file
st.store()
```

### Usage via CLI

> standardizer --input_file "tests/data/test_input.csv"






