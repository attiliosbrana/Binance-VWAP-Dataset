# Binance Portfolio Forecasting Hourly VWAP Dataset

The Binance Portfolio Forecasting Hourly VWAP Dataset provides a unique collection of historical cryptocurrency price data from Binance, the largest cryptocurrency exchange platform. The dataset includes data for 506 unique cryptocurrencies denominated in USD and has been preprocessed to calculate the hourly volume-weighted average price (VWAP) for each cryptocurrency. The dataset is useful for time series forecasting as VWAP is a more relevant indicator than just raw data points. The VWAP is calculated using the formula: ∑(price * volume) / ∑(volume). The dataset also encompasses many currency pairs for the VWAP calculation that are not available elsewhere. These series represent better, longer, and more consistent data series on historical currency prices. For example, some currencies were only traded against ETH for some time, and only years later started to trade against USD-equivalent pairs. This dataset manages to regain those USD-equivalent volume-weighted prices to train models on the true volatility of these cryptos. The dataset has been pre-processed, including forward-filling missing data points to maintain the continuity of the time series and prevent any disruptions in downstream analysis.

The dataset provides a Python dictionary serialized using the pickle format. It includes a 2D numpy array representing the historical hourly VWAP prices for each cryptocurrency, a 1D numpy array representing the tickers of the 506 cryptocurrencies in the series array, a 1D numpy array representing the dates in Unix time (milliseconds) of each hourly observation in the series array, a 2D numpy array representing the metadata of dates of hour, weekday, and day of the month, respectively, and several other useful elements for the dataset.

Overall, this dataset is a unique and valuable resource for researchers and practitioners interested in cryptocurrency trading, machine learning, and time series analysis. It provides a standardized, pre-processed, and consistent data series for over 500 cryptocurrencies, encompassing many currency pairs for the VWAP calculation that are not available elsewhere, enabling users to train more accurate and robust models for cryptocurrency price forecasting.

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

To load the dataset, you can use the following Python code:

```python
import pickle

# Load the dataset from file
with open('binance_dataset.pkl', 'rb') as f:
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

## Documentation version info
Python version: 3.10.8 | packaged by conda-forge | (main, Nov 22 2022, 08:26:04) [GCC 10.4.0]

Pickle version 4.0
