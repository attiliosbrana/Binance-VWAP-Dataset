import numpy as np
from numba import jit, njit
from datetime import datetime

@njit
def normalize(asts):
    """
    Function to normalize data using the last data.

    Parameters
    ----------
    asts : array
        a array (nxm) with float

    Returns
    ------
    array
        a array (nxm) with float
    """
    return (asts.T / asts[:, -1]).T

@njit
def weighted_port(asts, weights):
    """
    Function to calculate the value of each assets with it weights.

    Parameters
    ----------
    asts : array
        a array (nxm) with float
    weights : float/list
        a float or a list of float

    Returns
    -------
    array
        a array from the scalar product between asts and weights
    """
    return (asts.T * weights).sum(axis = 1)

@njit
def clean_up_weights(weights, threshold = 0.025):
    """
    Function to remove the assets from portfolio that has less then 2.5% representation
    of the total portfolio.

    Parameters
    ----------
    weights : float/list
        a float or a list of float
    threshold : float
        a threshold valeu

    Returns
    -------
    array
        a array (nxm) without the weights less then threshold
    """
    weights[weights < threshold] = 0
    return weights / weights.sum()

@njit
def random_weights(num_of_tokens):
    """
    Function to randomly set asset weights using Dirichlet distribution.

    Parameters
    ----------
    num_of_tokens : int
        a number of tokens

    Returns
    -------
    array
        a array with asset weights
    """
    return np.random.dirichlet(np.ones(num_of_tokens))

@njit
def train_window(dates_array, eligibles, train_cutoff, start, end, insample_size, outsample_size):
    """
    Function to divide the train data based on date. 

    Parameters
    ----------
    dates_array : array
        a array with hour, weekday and day
    eligibles : array
        the values before the cutoff date
    train_cutoff : int
        the value of cutoff date
    start : array
        the values of oldest_date
    end : array
        the values of newest_date
    insample_size : int
        a defined value for insample size
    outsample_size : int
        a defined value for outsample size
    
    Returns
    -------
    int
        values to represent the index and date
    array
        datetime in an array
    
    """
    main_token_idx = choice(eligibles)
    token_start, token_finish = start[main_token_idx], end[main_token_idx]
    min_start, max_end = token_start + 1, min(train_cutoff, token_finish) + 1
    cut_point = np.random.randint(min_start, max_end)
    backcast_start, backcast_end = max(token_start, cut_point - insample_size), cut_point
    forecast_start, forecast_end = cut_point, min(max_end, cut_point + outsample_size)
    date = dates_array[cut_point - 1]
    return main_token_idx, cut_point, backcast_start, backcast_end, forecast_start, forecast_end, date

@njit
def test_window(dates_array, eligibles, train_cutoff, start, end, insample_size, outsample_size):
    """
    Function to divide the teste data based on date. 

    Parameters
    ----------
    dates_array : array
        a array with hour, weekday and day
    eligibles : array
        the values before the cutoff date
    train_cutoff : int
        the value of
    start : array
        the values of oldest_date
    end : array
        the values of newest_date
    insample_size : int
        a defined value for insample size
    outsample_size : int
        a defined value for outsample size

    Returns
    -------

    int
        values to represent the index and date
    array
        datetime in an array

    """
    main_token_idx = choice(eligibles)
    token_start, token_finish = start[main_token_idx], end[main_token_idx]
    min_start, max_end = max(token_start, train_cutoff) + 1, token_finish + 1
    cut_point = np.random.randint(min_start, max_end)
    backcast_start, backcast_end = max(token_start, cut_point - insample_size), cut_point
    forecast_start, forecast_end = cut_point, min(max_end, cut_point + outsample_size)
    date = dates_array[cut_point - 1]
    return main_token_idx, cut_point, backcast_start, backcast_end, forecast_start, forecast_end, date

@njit
def tokens(shape):
    """
    Function to randomly select the quantity of assets.

    Parameters
    ----------
    int
        the amount of compatibles assets

    Returns
    -------
    int
        a random number of tokens
    """
    return np.random.randint(1, max(min([41, shape]), 2))

@njit
def idxs(compatibles, num_of_tokens):
    return np.random.choice(compatibles, num_of_tokens, replace = False)

@njit
def idxs(main_token_idx, compatibles, num_of_tokens):
    """
    Function to generate a array of random values based on the number of 
    compatible assets and tokens.

    Parameters
    ----------
    main_token_idx : int
        a random value generate by choice function
    compatibles : array
        a array of compatible assets
    num_of_tokens : int
        a random asset number

    Returns
    -------
    array
        a array of random values
    """
    tokens = np.random.choice(compatibles, num_of_tokens, replace = False)
    return np.append(tokens, main_token_idx)

@njit
def choice(x):
    """
    Function to generate a random sample from a given array.

    Parameters
    ----------
    x : int
        the values before the cutoff date

    Returns
    -------
    ndarray
        the probabilities associated with each entry in a.
    """
    return np.random.choice(x)

@njit
def slices(data, i, start, end):
    """
    Function to quickly slice a array.

    Parameters
    ----------
    data : dict
        a dict with asset data
    i : int
        a index value for desireble token
    start : int
        a value for star date
    end : int
        a value for end date

    Returns
    -------
    dictionary
        a dictionary with asset data
    """
    return data[i, start:end]

@njit
def append(a, b):
    """
    Function to quickly append a data to an array.
    
    Parameters
    ----------
    a : array
        a array where the following parameter will be stored
    b : int
        a value to be stored

    Returns
    -------
    array    
        a final array with the 'b' parameter stored in 'a'
    """
    return np.append(a, b)

@njit
def build_random_portfolios(main_token_idx, compatibility, series, backcast_start, backcast_end,
                     forecast_start, forecast_end, date):
    """
    Function to generate a collection of random portfolios with time efficency,
    based on the previous functions.

    Parameters
    ----------
    main_token_idx : int
        a value to represent token index
    compatibility : list
        the region where exist compatibility with it pair
    series : array
        the vwap values
    backcast_start : int
        the start date for backcast
    backcast_end : int
        the end date for backcast
    forecast_start : int
        the start date for forecast
    forecast_end : int
        the end date for forecast
    date : int
        an array with hour, weekday and day

    Returns
    -------
    int
        an int value for date
    """
    compatibles = compatibility[main_token_idx]
    num_of_tokens, tokens_idx = random_assets(compatibles, main_token_idx)
    
    backcasts = slices(series, tokens_idx, backcast_start, backcast_end)
    normalized_backcasts = normalize(backcasts)

    forecasts = slices(series, tokens_idx, forecast_start, forecast_end)
    normalized_forecasts = (forecasts.T / backcasts[:, -1]).T

    weights = clean_up_weights(random_weights(num_of_tokens + 1))

    portfolio_backcast = weighted_port(normalized_backcasts, weights)
    portfolio_forecast = weighted_port(normalized_forecasts, weights)
    
    portfolio_backcast = append(portfolio_backcast, date)
    return portfolio_backcast, portfolio_forecast

@njit
def random_assets(compatibles, main_token_idx):
    """
    Function to select a random asset based on the compatible ones.

    Parameters
    ----------
    compatibles : array
        an array of compatible assets
    main_token_idx : int
        a random value generate by choice function

    Returns
    -------
    int
        an int value for tokens
    """
    num_of_tokens = tokens(compatibles.shape[0])
    tokens_idx = idxs(main_token_idx, compatibles, num_of_tokens)
    return num_of_tokens, tokens_idx

@njit
def get_train_batch(dates, dates_array, train_cutoff, series, start_offsets, finish_offsets, eligibles,
                    compatibility, insample_size, outsample_size, batch_size):
    """
    Create the train batch for the random porftolios.
    Basically, this function has two main focus:
    Select the period where the backcast and forecast will start and end;
    Create the random porftolios for backcast and forecast.

    Parameters
    ----------
    dates : array
        an array with the sum of dates
    dates_array : array
        an array with hour, weekday and day
    train_cutoff : int
        the value of
    series : array
        the vwap values
    start_offsets : array
        the values of oldest_date
    finish_offsets : array
        the values of newest_date
    eligibles : array
        the values before the cutoff date
    compatibility : list
        the region where exist compatibility with it pair
    insample_size : int
        a defined value for insample size
    outsample_size : int
        a defined value for outsample size
    batch_size : int
        a defined value for batch size

    Returns
    -------
    array
        four arrays generated to create the samples
    """
    # Initiate batch
    date_array_len = 3
    insample = np.zeros((batch_size, insample_size + date_array_len))
    insample_mask = np.zeros((batch_size, insample_size + date_array_len))
    outsample = np.zeros((batch_size, outsample_size))
    outsample_mask = np.zeros((batch_size, outsample_size))
    
    for i in range(batch_size):

        # Get a window
        main_token_idx, cut_point, backcast_start, backcast_end, forecast_start, forecast_end, date = \
        train_window(dates_array, eligibles, train_cutoff, start_offsets, finish_offsets,
                     insample_size, outsample_size)
        
        # Build random portfolios
        portfolio_backcast, portfolio_forecast = build_random_portfolios(main_token_idx, compatibility,
                                                                         series, backcast_start, backcast_end,
                                                                         forecast_start, forecast_end, date)
        # Modify the batch array
        insample[i, -len(portfolio_backcast):] = portfolio_backcast
        insample_mask[i, -len(portfolio_backcast):] = 1.0
        outsample[i, :len(portfolio_forecast)] = portfolio_forecast
        outsample_mask[i, :len(portfolio_forecast)] = 1.0
    
    return insample, insample_mask, outsample, outsample_mask

@njit
def get_test_batch(dates, dates_array, train_cutoff, series, start_offsets, finish_offsets, eligibles,
                    compatibility, insample_size, outsample_size, batch_size):
    """
    Create the test batch for the random porftolios.
    Basically, this function has two main focus:
    Select the period where the backcast and forecast will start and end;
    Create the random porftolios for backcast and forecast.

    Parameters
    ----------
    dates : array
        an array with the sum of dates
    dates_array : array
        an array with hour, weekday and day
    train_cutoff : int
        the value of
    series : array
        the vwap values
    start_offsets : array
        the values of oldest_date
    finish_offsets : array
        the values of newest_date
    eligibles : array
        the values before the cutoff date
    compatibility : list
        the region where exist compatibility with it pair
    insample_size : int
        a defined value for insample size
    outsample_size : int
        a defined value for outsample size
    batch_size : int
        a defined value for batch size
    
    Returns
    -------
    array
        four arrays generated to create the samples
    """
    #Initiate batch
    date_array_len = 3
    insample = np.zeros((batch_size, insample_size + date_array_len))
    insample_mask = np.zeros((batch_size, insample_size + date_array_len))
    outsample = np.zeros((batch_size, outsample_size))
    outsample_mask = np.zeros((batch_size, outsample_size))
    
    for i in range(batch_size):

        #Get a window
        main_token_idx, cut_point, backcast_start, backcast_end, forecast_start, forecast_end, date = \
        test_window(dates_array, eligibles, train_cutoff, start_offsets, finish_offsets,
                     insample_size, outsample_size)
        
        #Build random portfolios
        portfolio_backcast, portfolio_forecast = build_random_portfolios(main_token_idx, compatibility,
                                                                         series, backcast_start, backcast_end,
                                                                         forecast_start, forecast_end, date)
        #Modify the batch array
        insample[i, -len(portfolio_backcast):] = portfolio_backcast
        insample_mask[i, -len(portfolio_backcast):] = 1.0
        outsample[i, :len(portfolio_forecast)] = portfolio_forecast
        outsample_mask[i, :len(portfolio_forecast)] = 1.0
    
    return insample, insample_mask, outsample, outsample_mask