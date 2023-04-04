# Binance Portfolio Forecasting Hourly VWAP Dataset & Codebase

The Binance Portfolio Forecasting Hourly VWAP Dataset is a collection of historical cryptocurrency price data from Binance, the largest cryptocurrency exchange platform. The dataset includes data for 506 unique cryptocurrencies denominated in USD and has been pre-processed to maintain the continuity of the time series and prevent any disruptions in downstream analysis. The dataset is available for download in the OSF database [1] and has been used in the N-BEATS Perceiver paper [2].

## Objective

This repository aims to provide a standard, pre-processed, and consistent data series for more than 500 cryptocurrencies. This resource intends to enable users to train more precise and robust algorithms for predicting the price of cryptocurrencies. The repository is organized into multiple folders, with each directory containing code and data for a specific dataset-related activity. In particular, it enables users to:

-   replicate the exact same dataset,
-   update the dataset for new data,
-   provide tools for using the dataset for sampling random portfolios with proper indexing so these can be used in different applications, such as machine learning or portfolio optimization, and
-   compare new models with the models in the N-BEATS Perceiver paper, using the exact same dataset of over 4 million test samples.


## Directory Structure

The repository is organized into several directories, each with its own detailed README file. These directories are:

### benchmarking/

This directory contains code for benchmarking new methods against the test set used in the paper N-BEATS Perceiver. It includes a Jupyter notebook `benchmark_example.ipynb` for testing user-defined models against the test files, as well as a Python module `metrics_fast.py` containing functions for calculating various metrics used in time series analysis.

### download_data/

This directory contains code for regularly updating cryptocurrency data from Binance and preprocessing it as VWAPs. The code includes several Python scripts for downloading data and getting available denominations for each cryptocurrency. The scripts in this directory allow users to download and preprocess the data themselves, thereby ensuring the dataset is up to date with the latest available data.

### pre-processing/

This directory contains code for preprocessing raw data from Binance into VWAPs. The code includes several Python scripts for calculating VWAPs for various cryptocurrencies and generating training and testing data for cryptocurrency portfolio forecasting. The scripts in this directory enable users to preprocess the raw data into VWAPs and to generate training and testing data in a format that is compatible with the N-BEATS Perceiver paper.

### sampler/

This directory contains optimized functions for normalizing a random portfolio to the index of 1 at the last observation using vector operations under the Numba library. The directory includes a Jupyter notebook `sample_notebook.py` for implementing the functions and generating training and testing data for cryptocurrency portfolio forecasting. The scripts in this directory allow users to normalize a random portfolio to the index of 1 and generate training and testing data for cryptocurrency portfolio forecasting.

Each directory contains a README file with more detailed information about the code and data it contains.

## The Binance VWAP Dataset

The dataset can be downloaded manually at the OSF (https://osf.io/fjsuh/) repository under the name `binance_dataset_original_20220112.pkl` or fetched with the file in `/sampler/sample_notebook.ipynb`.

The dataset provides a unique and valuable resource for researchers and practitioners interested in cryptocurrency trading, machine learning, and time series analysis. The dataset is useful for time series forecasting as volume-weighted average price (VWAP) is a more relevant indicator than just raw data points. The VWAP is calculated using the formula: ∑(price * volume) / ∑(volume). The dataset also encompasses many currency pairs for the VWAP calculation that are not available elsewhere. These series represent better, longer, and more consistent data series on historical currency prices. For example, some currencies were only traded against ETH for some time, and only years later started to trade against USD-equivalent pairs. This dataset manages to regain those USD-equivalent volume-weighted prices to train models on the true volatility of these cryptos. The dataset has been pre-processed, including forward-filling missing data points to maintain the continuity of the time series and prevent any disruptions in downstream analysis.

The dataset is provided as a Python dictionary serialized using the pickle format. It includes a 2D numpy array representing the historical hourly VWAP prices for each cryptocurrency, a 1D numpy array representing the tickers of the 506 cryptocurrencies in the series array, a 1D numpy array representing the dates in Unix time (milliseconds) of each hourly observation in the series array, a 2D numpy array representing the metadata of dates of hour, weekday, and day of the month, respectively, and several other useful elements for the dataset.

Table: Periods and Assets used in the study

| Data      | Period                     | Assets             | Source                |
|-----------|----------------------------|--------------------|-----------------------|
| Train set | Jul 14, 2017 - Jun 30, 2022 | 500 unique tickers | Binance historical data |
| Test set  | Jul 1, 2022 - Oct 31, 2022  | 387 unique tickers | Binance historical data |

## Pre-processing that generated this data
- The dataset was collected from the Binance Vision website: [https://data.binance.vision/](https://data.binance.vision/).
-   The dataset includes data for 506 unique cryptocurrencies denominated in USD, with different suffixes, such as USDT, USD, USDC, BUSD, BTC, ETH, BNB.
-   The prices of all assets are converted to USD using the most recent available exchange rates at the time of data collection.
-   Missing data points are forward-filled to ensure that there are no gaps in the dataset. This is done to maintain the continuity of the time series and prevent any disruptions in downstream analysis.
-   The hourly data for each cryptocurrency includes the average price and volume for a given period.
-   The average price is calculated based on the Open, High, Low, and Close prices for that period.
-   The dataset is preprocessed to calculate the hourly volume-weighted average price (VWAP) for each cryptocurrency, which is a common indicator used in technical analysis.
-   The VWAP is calculated using the formula: ∑(price * volume) / ∑(volume).
-   The dataset also provides the oldest date, newest date, and length of the data available for each cryptocurrency.

## Data Structure

The dataset is provided as a Python dictionary serialized using the pickle format. The keys and values in the dictionary are as follows:

-   `series`: a 2D numpy array of shape (506, 46460) representing the historical hourly VWAP prices for each cryptocurrency. Each row corresponds to a different cryptocurrency, and each column corresponds to a different hour. The prices are denominated in USD.
-   `tickers`: a 1D numpy array of shape (506,) representing the tickers of the 506 cryptocurrencies in the series array.
-   `dates`: a 1D numpy array of shape (46460,) representing the dates in Unix time (milliseconds) of each hourly observation in the series array.
-   `dates_array`: a 2D numpy array of shape (46460, 3) representing the metadata of `dates` of hour, weekday, and day of the month, respectively.
-   `train_cutoff`: an integer representing the Unix time (milliseconds) cutoff date for the end of the training period. Any observations before this date can be used for training, and any observations after this date should be used for testing.
-   `start_offsets`: a 1D numpy array of shape (506,) representing the positional index in the `series` array where each cryptocurrency's history starts.
-   `finish_offsets`: a 1D numpy array of shape (506,) representing the positional index in the `series` array where each cryptocurrency's history ends.
-   `dataset_sizes`: a 1D numpy array of shape (506,) representing the length of each cryptocurrency's history in the `series` array.
-   `compatibility`: a Python list of length 506, where each element is a list of tickers that are compatible with the corresponding ticker throughout its entire history. A ticker is considered compatible if it was not delisted or delisted midway.
-   `train_eligible`: a 1D numpy array representing the indices of the cryptocurrencies in the `series` array that are eligible for use in the training period, i.e., they existed during the training period.
-   `test_eligible`: a 1D numpy array representing the indices of the cryptocurrencies in the `series` array that are eligible for use in the testing period, i.e., they existed during the testing period.

## Usage

To load the dataset, users can use the following Python code:

```python
import pickle

# Load the dataset from file
with open('binance_dataset_original_20220112.pkl', 'rb') as f:
    data = pickle.load(f)

# Access the different elements of the dataset
series = data['series']
tickers = data['tickers']
dates = data['dates']
dates_array = data['dates_array']
train_cutoff = data['train_cutoff']
start_offsets = data['start_offsets']
finish_offsets = data['finish_offsets']
dataset_sizes = data['dataset_sizes']
compatibility = data['compatibility']
train_eligible = data['train_eligible']
test_eligible = data['test_eligible']

```

## Setting up the Conda environment

This project uses a Conda environment to manage dependencies. To set up the environment, follow the steps below:

1.  Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if you haven't already.
    
2.  Clone the GitHub repository to your local machine:
    
    ```bash
    
    git clone https://github.com/attiliosbrana/Binance-VWAP-Dataset.git
    ```
    
    Replace `your_username` and `your_repository` with the appropriate values.
    
3.  Navigate to the repository's directory:
    
    ```bash
    
    cd your_repository
    ```
    
    Replace `your_repository` with the appropriate value.
    
4.  Create the Conda environment using the `environment.yml` file:
    
    ```bash
    
    conda env create -f environment.yml
    ```
    
    This command will create a new Conda environment named `binance_vwap_env` and install the required dependencies.
    
5.  Activate the new environment:
    
    -   On Windows:
        
        ```bash
        
        conda activate binance_vwap_env
        ```
        
    -   On macOS and Linux:
        
        ```bash
        
        source activate binance_vwap_env
        ```
        
    
    You should now see the environment name in the command prompt/terminal.
    
6.  You can now run the project's scripts and notebooks in this environment.
    

To deactivate the Conda environment, use the following command:

```bash

conda deactivate
```

## Documentation version info
Python version: 3.10.8 | packaged by conda-forge | (main, Nov 22 2022, 08:26:04) [GCC 10.4.0]

Pickle version 4.0

## How to cite this

If you use this dataset in your research, please cite it using the following citation:

Sbrana, A. (2023). Binance VWAP Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7749449

You may also want to specify the date you accessed the dataset, as it is subject to updates:

Accessed on [Month Day, Year].

## References

- [1] Sbrana, A., & Lima de Castro, P. A. (2023, February 23). N-BEATS Perceiver: A Novel Approach for Robust Cryptocurrency Portfolio Forecasting (Version 1) [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-2618277/v1
- [2] Sbrana, A. (2023, March 18). Binance Portfolio Forecasting Hourly VWAP Dataset. https://doi.org/10.17605/OSF.IO/FJSUH
- [3] Sbrana, A., Pires, G. (2023, March 18). Binance-VWAP-Dataset. GitHub, Zenodo. https://doi.org/10.5281/zenodo.7749449
