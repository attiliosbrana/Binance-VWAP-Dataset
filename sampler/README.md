# Binance-VWAP-Dataset/sampler

The `sampler` folder contains two files, `sample_notebook.py` and `sampler.py`. 

`sampler.py` contains optimized functions that normalize a random portfolio to the index of 1 at the last observation using vector operations under the Numba library.

The `sample_notebook.py` file provides an example implementation of the functions in `sampler.py` to demonstrate their performance. It imports `get_train_batch` and `get_test_batch` from `sampler.py` and uses them to process a pre-processed dataset that includes the hourly volume-weighted average price (VWAP) for each cryptocurrency in Binance. 

## Dataset
The dataset used in `sample_notebook.py` is the Binance Portfolio Forecasting Hourly VWAP Dataset, which provides historical cryptocurrency price data from Binance. It includes data for 506 unique cryptocurrencies denominated in USD and has been pre-processed to calculate the hourly volume-weighted average price (VWAP) for each cryptocurrency. The dataset is useful for time series forecasting as VWAP is a more relevant indicator than just raw data points. The dataset has been pre-processed, including forward-filling missing data points to maintain the continuity of the time series and prevent any disruptions in downstream analysis.

## Functions
The `train_batch` and `test_batch` functions in `sample_notebook.py` run the `get_train_batch` and `get_test_batch` functions in `sampler.py` using specified hyperparameters. These functions generate training and testing data for cryptocurrency portfolio forecasting.

## Performance Tests
To test the performance of the implementation, we run the `train_batch` function for training data on 10 and 1000 samples and time the results. The performance of these functions is evaluated using the `%%timeit` magic command.

## Visualization
Finally, we provide a visualization of the insample and outsample outputs for 10 random portfolios using the generated training data. The resulting plots show the VWAP over time for the selected cryptocurrencies.

# File Descriptions

- `sample_notebook.py`: an example implementation of the functions in `sampler.py` to demonstrate their performance.
- `sampler.py`: contains optimized functions that normalize a random portfolio to the index of 1 at the last observation using vector operations under the Numba library.