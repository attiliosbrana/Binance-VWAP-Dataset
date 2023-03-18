# Importing all necessary libraries
import numpy as np
import pickle
import datetime
import os
import sys
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
from numba import jit, njit
from numba.typed import List
from train_funcs import *
from sampler import *

# Importing module from another folder
sys.path.append('../data_download/')
from utils import cols 

def main():
#get todays date
    today = datetime.today().strftime('%Y%m%d')

    # Set train_cutoff to 2022-07-01, as timestamp
    train_cutoff = np.datetime64('2022-07-01 01:00:00')

    # Assets pickle file path
    assets_path = './'

    # Date limit to slice the data
    date_limit = '2022-11-01'

    # Load pkl file called assets.pkl in the fileserver folder in one-liner
    assets = pickle.load(open(assets_path + 'assets.pkl', 'rb'))
    print('> Assets pickle file loaded.')

    # Sort dictionary alphabetically
    assets = dict(sorted(assets.items()))
    # On each asset vwap, remove all the dates above or equal date_limit
    for asset in assets:
        assets[asset]['vwap'] = assets[asset]['vwap'][assets[asset]['vwap'].index <= date_limit]

    # Check the size of all vwaps for all assets, if it is different from all others, return a list
    # with the index of the asset that has a different size
    for asset in assets:
        if assets[asset]['vwap'].shape[0] != assets['BTC']['vwap'].shape[0]:
            print(asset, 'has different size.')
    # Create dict for test
    data = create_dict(assets, train_cutoff)

    # Dump data to trainings_data.pkl
    pickle.dump(data, open('binance_dataset_' + today + '.pkl', 'wb'))
    print('> Training data pickle saved sucessfully')
    
if __name__ == '__main__':
    main()