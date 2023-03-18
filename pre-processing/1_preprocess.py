# Importing all necessary libraries
import numpy as np
import os
import sys
import pandas as pd
import pickle
from datetime import datetime
from os import listdir

# Importing module from another folder
sys.path.append('../data_download/')
from load_utils import * 
from utils import cols 

def main():

    # Defining list of Tickets
    denoms = ['USDT', 'USD', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']

    # Monthly file path
    monthly_path = '../data_download/data/spot/monthly/klines'

    # Desirable date for study
    first_date_ever = '2017-07-14 04:00:00'

    # Base assets pickle file path
    base_assets_path = '../data_download/'

    # Pickle file save path
    pickle_file_save_path = './'

    # Transform first_date_ever to timestamp
    first_date_ever = datetime.strptime(first_date_ever, '%Y-%m-%d %H:%M:%S')

    # Get every hour from first_date_ever to now
    all_hours = get_every_hour(first_date_ever)

    # Load pkl file called base_assets.pkl in the fileserver folder in one-liner
    base_assets = pickle.load(open(base_assets_path + 'base_assets.pkl', 'rb'))
    print('> base_assets pickle file loaded succesfully.')

    # Dictionary to store the VWAP of the convert assets, namely BTC, ETH, BNB, create empty keys
    convert_assets = {c: None for c in ['BTC', 'ETH', 'BNB']}

    # Dictionary to store the VWAP of the base assets, namely all the assets in base_assets
    # Include keys for vwap, oldest_date, newest_date, lenth, and avg_price_vwap
    assets = {c: {'vwap': None, 'oldest_date': None, 'newest_date': None, 'length': None, 'avg_price_df': None} for c in base_assets}

    # # for testing only: reduce assets to three random assets, choose three keys, include all convert assets
    # assets = {k: assets[k] for k in list(assets.keys())[:3] + list(convert_assets.keys())}

    # For each convert_asset, get the vwap, oldest_date, newest_date and length of the convert_asset,
    # through get_base_asset_vwap, and then store it in convert_assets
    for convert_asset in convert_assets.keys():
        assets = get_base_asset_vwap(convert_asset, assets, base_assets, cols, all_hours, convert_assets, monthly_path) #---> FAILS HERE
        convert_assets[convert_asset] = assets[convert_asset]['vwap']

        # Add the correlation between the vwap and the avg_price_df columns to the assets dictionary
        assets[convert_asset]['correlation'] = correlation_vwap_avg_price_df(convert_asset, assets)

        # Add mape_dict and diffs to assets dictionary
        assets[convert_asset]['mape_dict'], assets[convert_asset]['diffs'] = calculate_mape(convert_asset, assets)

    # Do the same as before for all assets that have a vwap of None
    for asset in assets.keys():
        try:
            if assets[asset]['vwap'] is None:
                assets = get_base_asset_vwap(asset, assets, base_assets, cols, all_hours, convert_assets, monthly_path)

                # Add the correlation between the vwap and the avg_price_df columns to the assets dictionary
                assets[asset]['correlation'] = correlation_vwap_avg_price_df(asset, assets)

                # Add mape_dict and diffs to assets dictionary
                assets[asset]['mape_dict'], assets[asset]['diffs'] = calculate_mape(asset, assets)
        except:
            print('Error with', asset)

    # Remove assets where VWAP is null
    for asset in list(assets.keys()):
        if assets[asset]['vwap'] is None:
            print(asset, 'has VWAP None.')
            # Remove the asset from the assets dictionary
            del assets[asset]

    # Check how many assets from the total have a non-null vwap
    print('Assets with non-null VWAP:', len([asset for asset in assets.keys() if assets[asset]['vwap'] is not None]))

    # Compare with the total of assets
    print('Number of assets:', len(assets.keys()))

    # Check if the sum of the vwap of the assets with a non-null vwap is larger than zero
    # steps:
    # 1. Get the assets with a non-null vwap
    # 2. Get the nansum of the vwap of the assets with a non-null vwap
    # 3. Check if the nansum is larger than zero
    # 4. Count the number of assets for which this is true
    print('Sum of assets with non-null VWAP:', len([asset for asset in assets.keys() if assets[asset]['vwap'] is not None and np.nansum(assets[asset]['vwap']) > 0]))

    # Save the assets dictionary as a pickle file
    pickle.dump(assets, open('./assets_test.pkl', 'wb'))
    print('> assets picke file saved succesfully.')

if __name__ == '__main__':
    main()