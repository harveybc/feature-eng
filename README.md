# Feature Engineering 

Plug-in based feature engineering operations, transform raw data to generate new data that better represent features so they improve the performance of a predictive model. __Work In Progress, NOT USABLE YET__.

[![Build Status](https://travis-ci.org/harveybc/feature-eng.svg?branch=master)](https://travis-ci.org/harveybc/feature-eng)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-feature-eng.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/feature-eng?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/feature-eng/blob/master/LICENSE)

## Description

Implements modular components for feature engineering, it can be expanded by installing plugins, it includes some plugins: a heuristic training signal generator, a MSSA decomposer and a MSSA predictor. 

All modules are usable both from command line and from class methods library.

## Installation

To install the package via PIP, use the following command:

> pip install -i https://test.pypi.org/simple/ feature-eng

Also, the installation can be made by clonning the github repo and manually installing it as in the following instructions.

### Github Installation Steps
1. Clone the GithHub repo:   
> git clone https://github.com/harveybc/feature-eng
2. Change to the repo folder:
> cd feature-eng
3. Install requirements.
> pip install -r requirements.txt
4. Install python package (also installs the console command data-trimmer)
> python setup.py install
5. Add the repo folder to the environment variable PYTHONPATH
6. (Optional) Perform tests
> python setup.py test
7. (Optional) Generate Sphinx Documentation
> python setup.py docs

### Command-Line Execution

feature_eng is implemented as a console command:
> feature_eng --help

### Command-Line Parameters

* __--list_plugins__: Shows a list of available plugins.
* __--plugin <plugin_name>__: Loads a plugin to process an input dataset.
* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset to be processed with a plugin.
* __--output_file <filename>__: (Optional) Filename for the output dataset. Defaults to the input dataset with the .output extension.
* __--input_config_file <filename>__: (Optional) Imports an existing configuration file and processes a dataset with it.
* __--output_config_file <filename>__: (Optional) Filename for the output configuration containing rows trimmed in columns 0 and columns trimmed in column 1. Defaults 

## Examples of usage

The following examples show both the class method and command line uses for one module, for examples of other modules, please see the specific module´s documentation.

### Example: Usage via CLI to list installed plugins

> feature_eng --list_plugins

### Example: Usage via CLI to execute an installed plugin with its parameters

> feature_eng --plugin heuristic_ts --input_file "tests/data/test_input.csv"

### Example: Usage via Class Methods (HeuristicTS plugin)

TODO: PLUGIN LIST & IMPORT

```python
from feature-eng.heuristic_ts.heuristic_ts import HeuristicTS
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

## Pre-Installed Plugins

All the plugin modules and their CLI commands are installed with the feature-eng package, the following sections describe each module briefly and link to each module's basic documentation. 

Additional detailed Sphinix documentation for all modules can be generated in HTML format with the optional step 7 of the installation process, it contains documentation of the classes and methods of all modules in the feature-eng package. 

## Heuristic Training Signal Generator

Generates an ideal training signal for trading using EMA_fast forwarded a number of ticks minus current EMA_slow as buy signal.

See [heuristic_ts Readme](../master/README_heuristic_ts.md) for detailed description and usage instructions.

## Multivariate Singular Spectrum Analysis (MSSA) Decomposer. 

Performs MSSA decomposition, save the output dataset containing a configurable number of components per feature or the sum of a configurable number of components.

See [MSSA Decomposer Readme](../master/README_mssa_decomposer.md) for detailed description and usage instructions.

## MSSA Predictor

Performs MSSA prediction for a configurable number of forward ticks, save the .output dataset containing the prediction for a configurable number of channels or its sum.

See [MSSA Predictor Readme](../master/README_mssa_predictor.md) for detailed description and usage instructions.


## Plugin Creation

To create a plugin, there are two ways, the first one allows to install the plugin from an external python package using setuptools and is useful for testing your plugins, the second way is to add a new pre-installed plugin to the feature-eng package by making a pull request to my repo so i can review it and merge it. Both methods are described in the following sections.

### External Plugin Creation

The following procedure allows to create a plugin as a python package with setuptools, install it, verify that is installed and use the plugin.

1. Create a new package with the same directory structure of the [standardizer plugin example](../master/examples/standardizer/)
2. Edit the setup.py or setup.cfg and add your package name as a feature_eng plugin (with a correspondent plugin name) in the entry_points section as follows:
> setup(
>     ...
>     entry_points={'feature_eng.plugins': '<PLUGIN_NAME> = <YOUR_PACKAGE_NAME>'},
>     ...
> )
3. Install your package as usual
> python setup.py install
4. Verify that your plugin was registered
> feature_eng --list_plugins
Check that <PLUGIN_NAME> appears in the list of installed plugins.
5. Use your newly installed plugin
> feature_eng --plugin <PLUGIN_NAME> --plugin_option1 --plugin_option2 ...

### Internal Plugin Creation

The following procedure allows to contribute to the feature_eng repository by creating a new plugin to be included in the pre-installed plugins.
1. Fork the feature_eng repository via the github homepage 
2. Clone your fork using github Desktop or via command line into a local directory
3. Create a new branch called with the name of the new plugin using github Desktop and select it
4. Cd to the feature_eng fork directory
5. Create the new module inside the plugins directory, following the structure of the existing plugins
7. Make a commit and push to save your changes to github
8. Make a Pull Request to the master branch of my feature_eng repo so i can review the changes and merge them with my existing code.

More detailed collaboration instructions soon.

