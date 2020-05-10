# FeatureEng: MSSA-Decomposer

Performs Multivariate Singular Spectrum Analysis (MSSA) decomposition of an input dataset.

[![Build Status](https://travis-ci.org/harveybc/feature_eng.svg?branch=master)](https://travis-ci.org/harveybc/feature_eng)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://harveybc-feature_eng.readthedocs.io/en/latest/)
[![BCH compliance](https://bettercodehub.com/edge/badge/harveybc/feature_eng?branch=master)](https://bettercodehub.com/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/harveybc/feature_eng/blob/master/LICENSE)

## Description

Performs MSSA decomposition of an input dataset, uses a configurable number of output channels, optionally 
grouping similar components.

## Installation

The plugin is pre-installed with the feature_eng package, the instructions are described in the [feature_eng README](../master/README.md).

### Command-Line Execution

The plugin's core method can be executed by loading the plugin by instantiating a FeatureEng class with plugin-specific configuration parameters and also, it can be used from the console command feature_eng, available after installing the [feature_eng package](../master/README.md):
> feature_eng --core_plugin mssa_decomposer --input_file <input_dataset> <optional_parameters>

### Command-Line Parameters

* __--input_file <filename>__: The only mandatory parameter, is the filename for the input dataset for the default feature_eng input plugin (load_csv).
* __--output_file <filename>__: (Optional) Filename for the output dataset for the default feature_eng output plugin (store_csv). Defaults to output.csv
* __--num_components <val>__:(Optional) Number of SSA components per input feature. Defaults to 0 = Autocalculated usign Singular Value Hard Thresholding (SVHT).
* __--group_similar <val>__: (Optional) If False, do not group similar components by adding them. Defaults to True.

## Examples of usage

The following example show how to configure and execute the core plugin.

```python
from feature_eng.feature_eng import FeatureEng
# configure parameters (same variable names as command-line parameters)
class Conf:
    def __init__(self):
        self.core_plugin = "mssa_decomposer"
        self.input_file = "tests/data/test_input.csv"
# initialize instance of the Conf configuration class
conf = Conf()
# initialize and execute the core plugin, loading the dataset with the default feature_eng 
# input plugin (load_csv), and saving the results using the default output plugin (store_csv). 
fe = FeatureEng(conf)
```







.






.