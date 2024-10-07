
# Feature Engineering System

## Description

The Feature Engineering System is a flexible, plugin-based tool designed for generating and selecting features from time-series data. This system allows for the integration of various feature engineering techniques via plugins, starting with the generation of technical indicators and supporting future methods such as Singular Spectrum Analysis (SSA) and Fast Fourier Transform (FFT).

### Key Features:

- **Plugin-Based Architecture**: The system uses a modular plugin architecture, allowing users to add, configure, and switch between different feature generation methods. The initial implementation includes a technical indicator generator, with support for additional plugins like SSA, FFT, and others in the future.
- **Configurable Parameters**: The system allows for dynamic configuration of input parameters such as input/output file paths, method-specific parameters, and other options via a command-line interface (CLI).
- **Correlation and Distribution Analysis**: Users can automatically compute and visualize Pearson and Spearman correlation matrices to identify relationships between features. The system also supports the visualization of feature distributions to help in manual feature selection.
- **Manual Feature Selection**: Users can manually select which features (e.g., technical indicators, SSA components) to include in the final output dataset, based on the results of the correlation and distribution analysis.

This tool is designed for data scientists, quantitative analysts, and machine learning practitioners working with time-series data in applications like financial modeling, trading strategies, and predictive analytics.

## Installation Instructions

To install and set up the feature-eng application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/feature-eng.git
    cd feature-eng
    ```

2. **Create and Activate a Virtual Environment**:

    - **Using `conda`**:
        ```bash
        conda create --name feature-eng-env python=3.9
        conda activate feature-eng-env
        ```

3. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Build the Package**:
    ```bash
    python -m build
    ```

5. **Install the Package**:
    ```bash
    pip install .
    ```

6. **(Optional) Run the feature-eng CLI**:
    - On Windows, verify installation:
        ```bash
        feature-eng.bat --help
        ```
    - On Linux:
        ```bash
        sh feature-eng.sh --help
        ```

7. **(Optional) Run Tests**:
    - On Windows:
        ```bash
        set_env.bat
        pytest
        ```
    - On Linux:
        ```bash
        sh ./set_env.sh
        pytest
        ```

## Usage

The application provides a command-line interface to control its behavior and manage feature generation through plugins.

### Command Line Arguments

#### Required Arguments

- `input_file` (str): Path to the input CSV file.

#### Optional Arguments

- `output_file` (str, optional): Path to the output CSV file. If not specified, the system will not generate an output file.
- `plugin` (str, default='technical_indicator'): Name of the plugin to use for feature generation. The default plugin generates technical indicators, but additional plugins such as SSA and FFT can be used.
- `correlation_analysis` (flag): Compute and display Pearson and Spearman correlation matrices.
- `distribution_plot` (flag): Plot the distributions of the generated features.
- `quiet_mode` (flag): Suppress output messages to reduce verbosity.
- `save_log` (str): Path to save the current debug log.
- `username` (str): Username for the remote API endpoint.
- `password` (str): Password for the remote API endpoint.
- `remote_save_config` (str): URL of a remote API endpoint for saving the configuration in JSON format.
- `remote_load_config` (str): URL of a remote JSON configuration file to download and execute.
- `remote_log` (str): URL of a remote API endpoint for saving debug variables in JSON format.
- `load_config` (str): Path to load a configuration file.
- `save_config` (str): Path to save the current configuration.

### Examples of Use

#### Generate Technical Indicators (default plugin)

To generate technical indicators using the default plugin:

```bash
f-eng.bat tests/data/eurusd_hour_2005_2020_ohlc.csv

```

#### Perform Singular Spectrum Analysis (plugin)

To perform SSA feature extraction:

```bash
f-eng.bat tests/data/eurusd_hour_2005_2020_ohlc.csv --plugin ssa

```

#### Fast Fourier Transform (plugin)

To perform FFT feature extraction:

```bash
f-eng.bat tests/data/eurusd_hour_2005_2020_ohlc.csv --plugin fft

```

#### Example with Distribution Plotting

Distribution plotting is enabled to visualize the distributions of the generated technical indicators, a normal distribution is recommended when using Pearson correlation (see Correlation Analysis), while a Spearman is recommended otherwise:

```bash
f-eng.bat tests/data/eurusd_hour_2005_2020_ohlc.csv --distribution_plot
```

#### Run Correlation Analysis

To compute and display correlation matrices for the generated features, use Pearson preferibly when the features have normal distribution, otherwise, use Spearman, both are calculated with the command:

```bash
f-eng.bat --input_file tests/data/eurusd_hour_2005_2020_ohlc.csv --correlation_analysis
```


## Project Directory Structure

```md
feature-eng/
│
├── app/                           # Main application package
│   ├── cli.py                     # Handles command-line argument parsing
│   ├── config.py                  # Stores default configuration values
│   ├── config_merger.py           # Merges configuration from various sources
│   ├── plugin_loader.py           # Dynamically loads feature engineering plugins
│   ├── data_handler.py            # Handles data loading and saving
│   ├── data_processor.py          # Processes input data and runs the feature extraction pipeline
│   ├── main.py                    # Main entry point for the application
│   └── plugins/                   # Plugin directory
│       ├── technical_indicator.py # Plugin for generating technical indicators
│       ├── ssa.py                 # Plugin for Singular Spectrum Analysis (future)
│       └── fft.py                 # Plugin for Fast Fourier Transform (future)
│
├── tests/                         # Test modules for the application
│   ├── system                     # System tests
│   └── unit                       # Unit tests
│
├── README.md                      # Project documentation
├── requirements.txt               # Python package dependencies
├── setup.py                       # Script for packaging and installing the project
├── set_env.bat                    # Batch script for environment setup
├── set_env.sh                     # Shell script for environment setup
└── .gitignore                     # Specifies untracked files to ignore
```

## Contributing

Contributions to the project are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
