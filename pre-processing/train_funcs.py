# Importing libraires
import numpy as np
import sys

sys.path.append('../sampler/')
from sampler import get_train_batch, get_test_batch

def compatibility(start_offsets, finish_offsets):
    """
    Function to check if the assets dates are compatible.

    Parameters
    ----------
    start_offsets : datetime
        a datetime object for the start date for assets comparation
    finish_offsets : datetime
        a datetime object for the finish date for assets comparation

    Returns
    -------
    list
        a list if compatible dates between assets
    """
    
    compatibility_list = []
    for i in range(finish_offsets.shape[0]):
        compatibles = (start_offsets[i] - start_offsets >= np.timedelta64(0)) & (finish_offsets[i] - finish_offsets <= np.timedelta64(0))
        compatibles = compatibles.nonzero()[0]
        compatibility_list.append(compatibles)
    return compatibility_list

def create_dict(assets, train_cutoff):
    """
    Function to create a dictionary with assets, start_offsets, finish_offsets and compatibility.

    Parameters
    ----------
    assets : dictionary
        a dict with assets data
    train_cutoff : datetime
        the date that is wanted to cut the historical data
    Returns
    -------
    dictionary
        a dictionary with assets, start_offsets, finish_offsets and compatibility data
    """
    
    # Create an empty dictionary to store the series, start_offsets, finish_offsets, compatibility_list
    dict = {}

    # Create an empty list to store the series, start_offsets, finish_offsets and lengths
    series = []
    start_offsets = []
    finish_offsets = []
    lengths = []

    # For each asset, append the series, start_offsets, finish_offsets, compatibility_list
    for asset in assets:
        series.append(assets[asset]['vwap'].values)
        start_offsets.append(assets[asset]['oldest_date'])
        finish_offsets.append(assets[asset]['newest_date'])
        lengths.append(assets[asset]['length'])

    # Concatenate the series
    series = np.concatenate(series, axis = 1)

    # Convert start_offsets, finish_offsets, compatibility_list to numpy arrays
    start_offsets = np.array(start_offsets)
    finish_offsets = np.array(finish_offsets)
    lengths = np.array(lengths)

    # Convert from numpy dates
    start_offsets = np.array([np.datetime64(start_offset) for start_offset in start_offsets]).astype(int)
    finish_offsets = np.array([np.datetime64(finish_offset) for finish_offset in finish_offsets]).astype(int)

    # Create a list to store the compatibility_list return
    compatibility_list = compatibility(start_offsets, finish_offsets)

    # Dates and train cutoff
    dates = assets[list(assets.keys())[0]]['vwap'].index
    dates_array = np.array((dates.hour.values, dates.weekday.values, dates.day.values)).T
    train_cutoff = int(dates[(dates == train_cutoff).nonzero()[0]].astype(int)[0]/10**3)
    dates = dates.astype(int)

    # Substitute all nan values with 0 in data['series']
    dict['series'] = np.nan_to_num(series)

    # Transpose the shape of the series
    dict['series'] = dict['series'].T 

    # Add the series, start_offsets, finish_offsets, compatibility_list to the dictionary
    dict['tickers'] = list(assets.keys())
    dict['dates'] = dates
    dict['dates_array'] = dates_array
    dict['train_cutoff'] = train_cutoff
    dict['start_offsets'] = start_offsets
    dict['finish_offsets'] = finish_offsets
    dict['dataset_sizes'] = lengths
    dict['compatibility'] = compatibility_list

    # Print train_cutoff and start_offsets
    # print(train_cutoff)
    # print(start_offsets)
    dict['train_eligible'] = np.where(dict['start_offsets'] <= train_cutoff)[0]
    dict['test_eligible'] = np.where(dict['finish_offsets'] >= train_cutoff)[0]

    # Fix dates to ints
    dict['dates'] = (dict['dates'] / 10**6).astype(int).to_numpy()
    dict['train_cutoff'] = int(dict['train_cutoff'] / 10**3)
    dict['train_cutoff'] = np.where(dict['dates'] == dict['train_cutoff'])[0][0]
    dict['start_offsets'] = (dict['start_offsets'] / 10**3).astype(int) ##??
    dict['finish_offsets'] = (dict['finish_offsets'] / 10**3).astype(int)

    # Transform start_offsets and finish_offsets from numpy dates to positional index using np.where relative to date['dates']
    dict['start_offsets'] = np.array([np.where(dict['dates'] == start_offset)[0][0] for start_offset in dict['start_offsets']])
    dict['finish_offsets'] = np.array([np.where(dict['dates'] == finish_offset)[0][0] for finish_offset in dict['finish_offsets']])

    return dict

# Rewrite the train_batch function outside of the class
def train_batch(data, insample_size, outsample_size, batch_size):
    """
    A simple way to run the 'get_train_batch' function in sampler using the hyperparameters.
    Parameters
    ----------
    data : dictionary
        a dict with assets values
    insample_size : int
        the value for insample
    outsample_size : int
        the value for outsample
    batch_size :
        the size of the batch
    Returns
    -------
    array
        four array with values for x and y as well for the masks
    """
    # Get_train_batch for self.data
    return get_train_batch(data['dates'], data['dates_array'], data['train_cutoff'],
                             data['series'],
                             data['start_offsets'], data['finish_offsets'],
                             data['train_eligible'], data['compatibility'],
                             insample_size, outsample_size, batch_size)
# Test batch outside of the class
def test_batch(data, insample_size, outsample_size, batch_size):
    """
    A simple way to run the 'get_test_batch' function in sampler using the hyperparameters.
    Parameters
    ----------
    data : dictionary
        a dict with assets values
    insample_size : int
        the value for insample
    outsample_size : int
        the value for outsample
    batch_size :
        the size of the batch
    Returns
    -------
    array
        four array with values for x and y as well for the masks
    """
    # Get_test_batch for self.data
    return get_test_batch(data['dates'], data['dates_array'], data['train_cutoff'],
                             data['series'],
                             data['start_offsets'], data['finish_offsets'],
                             data['test_eligible'], data['compatibility'],
                             insample_size, outsample_size, batch_size)

# def pkls_health(data, data_alt):
#     # Checking compatibility between assets data
#     print('-------- Pickles health --------')
#     print('Size of assets.pkl:', data['dates'].shape[0])
#     print('Size of data_numpy.pkl:', data_alt['dates'].shape[0])
#     print('Checking if both pkl start with same date:', data['dates'][0] == data_alt['dates'][0])
    
#     # Checking if the matrix of hour, weekday and day is the same for both dictionaries.
#     # When checking data dict slice the dict is needed.
#     print('Both plk has the same time matrix:', np.all(data_alt['dates_array'] == data['dates_array'][:data_alt['dates_array'].shape[0]]))

# def find_ticker(data, data_alt):
#     """
#     Function that randomly select a asset in data dict and check if the 
#     converted ticker (ticker + USDT) is in the data_alt dict.

#     Parameters
#     ----------
#     data : dict
#         a dictionary with asset data
#     data_alt : dict
#         a dictionary with asset data

#     Returns
#     -------
#     str
#         the ticker in data_alt dict
#     """
#     ticker = np.random.choice(data['tickers'])
#     print(ticker)
#     if ticker + 'USDT' in data_alt['USDT']['tickers']:
#         print('Found ticker', ticker)
#         return ticker
#     else:
#         return find_ticker(data, data_alt)
    
# def plot_series(ticker, data, data_alt):
#     """
#     Function to generate a plot between two dictionaries: data and data_alt

#     Parameters
#     ----------
#     ticker : str
#         the name of asset
#     data : dict
#         the dictionary with all valeus of the ticker
#     data_alta : dict
#         the dictionary with all valeus of the ticker
    
#     Returns
#     matplotlib.image
#     """
#     #Find the index in data_alt['USDT']['tickers'] of the ticker + 'USDT', using np.where
#     index = np.where(data_alt['USDT']['tickers'] == ticker + 'USDT')[0][0]
    
#     plt.figure(figsize=(8,4))
#     plt.plot(data['series'][data['tickers'].index(ticker)], label = 'data')
#     plt.plot(data_alt['USDT']['series'][index], label = 'data_alt')
#     plt.title('Comparisson Plot')
#     plt.xlabel('Time')
#     plt.ylabel('VWAP')
#     plt.legend()
#     plt.show()

# def plot_data(insample):
#     """
#     Function to generate a plot between two dictionaries: data and data_alt

#     Parameters
#     ----------
#     ticker : str
#         the name of asset
#     data : dict
#         the dictionary with all valeus of the ticker
#     data_alta : dict
#         the dictionary with all valeus of the ticker
    
#     Returns
#     matplotlib.image
#     """
#     #Find the index in data_alt['USDT']['tickers'] of the ticker + 'USDT', using np.where
#     index = np.where(data_alt['USDT']['tickers'] == ticker + 'USDT')[0][0]
    
#     plt.figure(figsize=(8,4))
#     plt.plot(insample)
#     plt.title('Insample Plot')
#     plt.xlabel('Time')
#     plt.ylabel('VWAP')
#     plt.show()

# seasonal_pattern = 'Hourly'
# lookback = Meta.lookbacks[0]

# Creating a class to store the parameters for training and test.
# class HyperParameters():
#     """
#     A class used to store the hyperparameters for training and testing.
#     """

#     def __init__(self):
#         """
#         A function to store and calculate the frequency parameters for trainig and testing.
#         """
#         #Meta date data
#         self.date_array_len = 3
        
#         #Frequency
#         self.seasonal_pattern = seasonal_pattern
#         self.lookback = lookback
#         self.outsample_size = Meta.horizons_map[self.seasonal_pattern]
#         self.insample_size = self.lookback * self.outsample_size
#         self.input_size = self.insample_size + self.date_array_len
#         self.timeseries_frequency = Meta.frequency_map[self.seasonal_pattern]
        
#         #Training params
#         self.epochs = 30000
#         self.retrain_epochs = 9000
#         self.batch_size = 1024
#         self.lr_decay_step = self.epochs // 3
#         self.learning_rate =  0.001

#         #Options
#         self.loss_options = ['MASE', 'SMAPE', 'MAPE']
        
#         #stack params: input_size, output_size, stacks, layers, layer_size
#         self.stacks = 30
#         self.layers = 4
#         self.layer_size = 512
#         self.stack_params = [self.input_size, self.outsample_size, self.stacks, self.layers, self.layer_size]