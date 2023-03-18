#Import all required libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from os import listdir
import os

def get_every_hour(first_date_ever):
    """
    Check if input parameter 'first_date_ever' is datetime and return every hour from specific date.

    Parameters
    ----------
    first_date_ever : datetime
        The selected date for training data.

    Raises
    ------
    TypeError
        Input parameter must be a datetime object.

    Returns
    -------
    list
        a list of datetime objects
    """
    # add isso para checar se o parametro tá correto, nsei se é útil
    if not isinstance(first_date_ever, datetime):
        raise TypeError("Input parameter must be a datetime object")

    now = datetime.now()
    #get now's month
    now_month = now.month
    #get the last hour of last month
    last_month_last_hour = datetime(now.year, now_month, 1)
    last_month_last_hour = datetime(2022, 11, 1)

    every_hour = []
    while first_date_ever < last_month_last_hour:
        every_hour.append(first_date_ever)
        first_date_ever += timedelta(hours = 1)
    return every_hour

def get_zips(asset, monthly_path):
    """
    Check if the asset zip file path exists, create a list of 
    files at that path and return a list for monthly asset files.

    Parameters
    ----------
    asset : str
        a single or a list of assets
    monthly_path : str
        the path location of monthly file

    Returns
    -------
    list
        a list string of monthly asset file
    """
    if os.path.exists(monthly_path + '/' + asset + '/1m'):
        monthly_zips = sorted([f for f in listdir(monthly_path + '/' + asset + '/1m') if 'CHECKSUM' not in f])
    else:
        monthly_zips = []
    return monthly_zips

def load_zips(asset, monthly_zips, cols, monthly_path):
    """
    Read all csv zip files that was located by 'get_zips' function for monthly data.
    Also rename the dataframe columns and concat every dataframe createad. 
    open time.

    Parameters
    ----------
    asset : str/list
        a single or a list of assets
    monthly_zips : list 
        a list of all assets monthly zip files
    cols : list
        a list of columns names
    monthly_path : str
        the path location of monthly file

    Returns
    -------
    dataframe
        a dataframe with all data from specific csv zip file, sorted by 'Open time'
    """
    dfs = []
    for z in monthly_zips:
        new_df = pd.read_csv(monthly_path + '/' + asset + '/1m/' + z, header = None, usecols = list(range(len(cols))))
        new_df.columns = cols
        dfs.append(new_df)
    df = pd.concat(dfs, ignore_index = True)
    df = df.astype({c: 'float32' for c in df.dtypes.index[df.dtypes == 'float64']})
    return df.sort_values('Open time', ignore_index = True)

def get_crypto_data(crypto, cols, all_hours, monthly_path):
    """
    Execute 'get_zips' and 'load_zips' functions and perform some data transformation:
    Convert 'Open Time' column to datetime in ms and set as index, calculate the average 
    price ('Avg Price' column) based on OHLC columns and filter the final dataframe 
    with 'Avg Price' and 'Volume' columns.
    
    Parameters
    ----------
    crypto : dict
        a dict of cryptocurrency
    cols : list
        a list of columns names
    all_hours : list
        a list of datetime object
    monthly_path : str
        the path location of monthly file

    Returns
    -------
    dataframe
        a dataframe with assets data
    """
    # Run functions
    monthly_zips = get_zips(crypto, monthly_path)
    df = load_zips(crypto, monthly_zips, cols, monthly_path)

    # Data Transformation
    df['Open time'] = pd.to_datetime(df['Open time'], unit = 'ms')
    df = df.set_index('Open time')

    df['Avg Price'] = df[['Open', 'High', 'Low', 'Close']].mean(axis = 1)

    df = df[['Avg Price', 'Quote asset volume']]
    df.columns = ['Avg Price', 'Volume']
    return df

def get_all_assets_merge(crypto, denoms, cols, all_hours, monthly_path):
    """
    The function accpets crypto and its denoms, and returns all 'Avg Prices' and 
    'Volume' from 'get_crypto_data' function, in separate dataframes.

    Parameters
    ----------
    crypto : dict
        a dict of cryptocurrency
    denoms : list
        a list of tickets 
    cols : list
        a list of columns names
    all_hours : list
        a list of datetime object
    monthly_path : str
        the path location of monthly file

    Returns
    -------
    dataframe
        two dataframes of average price and volume
    """
    df = pd.DataFrame(index = all_hours)
    
    #Iterate through denoms
    for denom in denoms:

        # Try, except to avoid errors, print crypto and denom that failed
        
        try:
            #Get crypto denom
            crypto_denom = crypto + denom
            
            #Get df
            new_df = get_crypto_data(crypto_denom, cols, all_hours, monthly_path)

            #Add a suffix to the column names with the denomination
            new_df.columns = [col + '_' + denom for col in new_df.columns]
            
            #Make left merge with df based on index
            df = df.merge(new_df, how = 'left', left_index = True, right_index = True)

        except:
            print(crypto, denom, 'failed')
            continue

    #Get Avg Price and Volume dfs
    avg_price_df = df[[col for col in df.columns if 'Avg Price' in col]]
    volume_df = df[[col for col in df.columns if 'Volume' in col]]

    return avg_price_df, volume_df

def calculate_vwap(avg_price_df, volume_df):
    """
    Provide the calculation for VWAP based on both dataframe returned by 
    'get_all_assets_merge' function: 'avg_price_df' and 'volume_df'.

    All NaNs values were considered as zero.

    Parameters
    ----------
    avg_price_df : dataframe
        a dataframe fulled by average price from selected assets
    volume_df : dataframe
        a dataframe fulled by volume from selected assets
    
    Returns
    -------
    dataframe
        a dataframe fulled by VWAP
    """

    #Calculate VWAP while transforming to numpy
    vwap = avg_price_df.values * volume_df.values

    #Make division ignoring nans, and avoiding RuntimeWarning: invalid value encountered in true_divide
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap = np.nansum(vwap, axis = 1) / np.nansum(volume_df.values, axis = 1)

    #Create df
    vwap_df = pd.DataFrame(vwap, index = avg_price_df.index, columns = ['VWAP'])

    return vwap_df

def get_oldest_newest_date(df):
    """
    Provide the newst and oldest date in which the dataframe is not null.

    Parameters
    ----------
    df : dataframe
        a dataframe o interest that has null values

    Returns
    -------
    datetime
        two datetime objects
    """
    df = df.index[~df.isnull().all(1)]
    oldest_date = df[0]
    newest_date = df[-1]
    return oldest_date, newest_date

def fill_missing_hours(df, all_hours, oldest_date, newest_date):
    """
    Guarantee that dataframe has all hours need in the specific period.
    If so, fill the NaNs values with the last valid observation.

    Parameters
    ----------
    df : dataframe
        a interest dataframe to be filled
    all_hours : list
        a list of datetime object
    oldest_date : datetime
        a datetime object for the oldest date with null values in dataframe 
    newest_date : datetime
        a datetime object for the newest date with null values in dataframe

    Returns
    -------
    datafarme
        a dataframe with filled NaNs
    """
    df = df.reindex(all_hours)
    df.loc[oldest_date:newest_date] = df.loc[oldest_date:newest_date].fillna(method = 'ffill')
    return df

def convert_dfs_to_usd(avg_price_df, volume_df, base_asset, base_asset_vwap):
    """
    Convert the 'avg_price_df' and 'volume_df' to USD using the VWAP of a TBD base asset.

    Parameters
    ----------
    avg_price_df : dataframe
        a dataframe with all average prices
    volume_df : dataframe
        a dataframe with all volume data
    base_asset : str
        the asset tickets
    base_asset_vwap : dict
        a dictionary of asset tickets and data
    Returns
    -------
    dataframe
        two dataframes converted
    """
    # Transforming to USD
    avg_price_df['Avg Price_' + base_asset + '_USD'] = avg_price_df['Avg Price_' + base_asset] * base_asset_vwap[base_asset]
    volume_df['Volume_' + base_asset + '_USD'] = volume_df['Volume_' + base_asset] * base_asset_vwap[base_asset]

    # Dropping columns
    avg_price_df = avg_price_df.drop('Avg Price_' + base_asset, axis = 1)
    volume_df = volume_df.drop('Volume_' + base_asset, axis = 1)

    return avg_price_df, volume_df

def get_crypto_data_vwap(crypto, denoms, cols, all_hours, convert_assets, monthly_path):
    """
    Function to get crypto data, by 'get_all_assets_merge', function and convert it to VWAP, 
    by 'calculate_vwap' function.

    Parameters
    ----------
    crypto : dict
        a dict of cryptocurrency
    denoms : list
        a list of tickets 
    cols : list
        a list of columns names
    all_hours : list
        a list of datetime object
    convert_assets : dict
        a dictionary with crypto tickets for converted data
    monthly_path : str
        the path location of monthly file

    Returns
    -------
    dataframe
        a calculated dataframe
    
    str
        the lenght of vwap_df
    
    datetime
        the oldest and newest date
    """
    #Get crypto data
    avg_price_df, volume_df = get_all_assets_merge(crypto, denoms, cols, all_hours, monthly_path)
    
    #Convert all columns that include a convert asset from convert_assets keys to USD,
    # if they exist as a suffix for any of the columns in avg_price_df and volume_df

    for convert_asset in convert_assets.keys():
        if 'Avg Price_' + convert_asset in avg_price_df.columns:
            avg_price_df, volume_df = convert_dfs_to_usd(avg_price_df, volume_df, convert_asset, convert_assets[convert_asset])

    #Calculate VWAP
    vwap_df = calculate_vwap(avg_price_df, volume_df)

    #Get the oldest and newest date in which the dataframe is not null
    oldest_date, newest_date = get_oldest_newest_date(vwap_df)

    #Fill in missing hours
    vwap_df = fill_missing_hours(vwap_df, all_hours, oldest_date, newest_date)

    #Get the length of non null values of the dataframe
    length = len(vwap_df.index[~vwap_df.isnull().all(1)])

    #Name the column after the crypto
    vwap_df.columns = [crypto]

    return vwap_df, oldest_date, newest_date, length, avg_price_df

def get_base_asset_vwap(base_asset, assets, base_assets, cols, all_hours, convert_assets, monthly_path):
    """
    Function to get the VWAP, 'oldest_date', 'newest_date' and length of a base asset, and store it in assets.

    Parameters
    ----------
    base_asset : dict
        a dictinory with assets tickets
    assets : list
        asset tickets from pickle file 
    base_assets : pkl
        a pickle file with all assets tickets
    cols : list
        a list of columns names
    all_hours : list
        a list of datetime object
    convert_assets : dict
        a dictionary with crypto tickets
    monthly_path : str
        the path location of monthly file
    
    Returns
    -------
    dataframe
        a assets dataframe
    """
    denoms = base_assets[base_asset]
    assets[base_asset]['vwap'], assets[base_asset]['oldest_date'], assets[base_asset]['newest_date'], \
        assets[base_asset]['length'], assets[base_asset]['avg_price_df'] = get_crypto_data_vwap(base_asset, denoms, cols, all_hours,\
             convert_assets, monthly_path)
    return assets

def plot_vwap_avg_price_df(asset, assets):
    """
    Function to compare an asset's vwap vs the avg_price_df that have a USDT, USD, USDC or BUSD suffix
    all are stored in the assets dictionary. Use a plot to show the comparison. Add a counter to each
    plot and add 100 progressively to each series plotted.

    Parameters
    ----------
    asset : str
        the asset name
    assets : dict
        a dictionary assets data
    
    Returns
    -------
    matplotlib.figure.Figure
    
    """
    vwap = assets[asset]['vwap']
    avg_price_df = assets[asset]['avg_price_df']

    #Plot the vwap
    vwap.plot(label = 'VWAP_' + asset, legend = True, secondary_y = True)

    counter = 1.5
    for suffix in ['USDT', 'USD', 'USDC', 'BUSD']:
        #If suffix is present in the string of any column in avg_price_df, plot it
        for col in avg_price_df.columns:
            #Check if the col ends with the suffix
            if col.endswith(suffix):
                (avg_price_df[col] * counter).plot(label = 'Avg Price_' + asset + '_' + suffix, legend = True, alpha = 0.5)
                counter += 0.5
    
    plt.show()

def correlation_vwap_avg_price_df(asset, assets):
    """
    Function to calculate Perason correaltion between VWAP and Average Price.
    If the correlation is less than 0.9 an alert is generated.

    Parameters
    ----------
    asset : str
        the asset name
    assets : dict
        a dictionary assets data

    Returns
    -------
    dataframe
        a correlation matrix
    string
        an alert 
    """
    vwap = assets[asset]['vwap']
    avg_price_df = assets[asset]['avg_price_df']

    #Add vwap to avg_price_df
    avg_price_df['VWAP_' + asset] = vwap
    
    #Calculate correlation
    corr = avg_price_df.corr(method = 'pearson', min_periods = 1)

    #Generate a string that alerts if correlation is less than 0.9, specify the assets, output the correlation value
    alert = None
    if corr.iloc[-1, :-1].min() < 0.9:
        alert = ['Alert: correlation is less than 0.9 for ' + asset + ' with ' + corr.iloc[-1, :-1].idxmin(), corr.iloc[-1, :-1].min()]

    return corr, alert

def divide_no_nan(a, b):
    """
    Function to remove all data that is NaN.
    Notice: when a/b result is NaN or Inf, it is replaced by 0.

    Parameters
    ----------
    a : array
        a array with numerical values
    b : array
        a array with numerical values

    Returns
    -------
    result
    float
        division result
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

def mape_loss_np(forecast, target, mask):
    """
    Calculate the mape loss function using numpy.

    Parameters
    ----------
    forecast : array
        a array of y_hat values
    target : array
        a array of y_true values
    mask : array
        a array of determined values with same shape as forecast

    Returns
    -------
    float
        mape value
    """
    weights = divide_no_nan(mask, target)
    return np.nanmean(np.abs((forecast - target) * weights)) * 100

def calculate_mape(asset, assets):
    """
    Function to calculate de MAPE between VWAP and Averga Price of a given asset.

    Parameters
    ----------
    asset : str
        the asset name
    assets : dict
        a dictionary assets data

    Returns
    -------
    dict
        dictionary with mape values
    
    diffs
        list with vwap and average price differences
    """
    vwap = assets[asset]['vwap']
    avg_price_df = assets[asset]['avg_price_df']

    #Create a dictionary to store the MAPE values
    mape_dict = {}
    diffs = []

    #For each column in avg_price_df, calculate the MAPE, add ones like vwap
    for col in avg_price_df.columns:
        diff_a = vwap.values.flatten()
        diff_b = avg_price_df[col].values.flatten()
        diff = diff_a - diff_b

        #Round the MAPE to 3 decimal places
        mape_dict[col] = round(mape_loss_np(diff_a, diff_b, np.ones_like(vwap)), 3)

        #Add index to diff
        diff = pd.DataFrame(diff, index = vwap.index, columns = [col])

        #If vwap not in col, append to diffs
        if 'VWAP' not in col:
            diffs.append(diff)

        #Plot the diffs, add legend
        # plt.plot(diff, label = col)
        # plt.legend()

    #Concatenate the diffs
    diffs = pd.concat(diffs, axis = 1)
    
    #Print the mape_dict
    # print(mape_dict)

    return mape_dict, diffs

def plot_diffs(asset, assets):
    """
    Function to plot the diffs of any asset.

    Parameters
    ----------
    asset : str
        the asset name
    assets : dict
        a dictionary assets data

    Returns
    -------
        matplotlib.figure.Figure
    """
    diffs = assets[asset]['diffs']
    plt.plot(diffs)
    plt.legend(diffs.columns)
    plt.show()

def print_report(asset, assets):
    """
    Rerpot function that shows:
    Oldest date, Newest date, Lenght of dataframe, Correlattion between VWAP and Average Price,
    MAPE, Plot of assets diff and Plot of VWAP vs Average Price.

    Parameters
    ----------
    asset : str
        the asset name
    assets : dict
        a dictionary assets data

    Returns
    -------
    matplotlib.figure.Figure
    
    """
    print('Oldest date ------------------->', assets[asset]['oldest_date'])
    print('Newest date ------------------->', assets[asset]['newest_date'])
    print('Length ------------------------>', assets[asset]['length'])
    print('Correlation Matrix ------------:')
    print(assets[asset]['correlation'][0])
    #if there is an alert message different from none, print it
    if assets[asset]['correlation'][1] != None:
        print(assets[asset]['correlation'][1])
    print('MAPE --------------------------:')
    print(assets[asset]['mape_dict'])
    print('\n')
    print('MAPE Diffs --------------------:')
    plot_diffs(asset, assets)
    print('Dislocated VWAP vs Avg Price DF:')
    plot_vwap_avg_price_df(asset, assets)
    #print space
    print('')