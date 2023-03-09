# Import necessary modules
from glob import glob
from pathlib import Path
from time import sleep
import pickle
import random
from datetime import *
from enums import *
from utility import get_all_symbols, get_parser, get_path
from utils import get_last_closed_month
from os import listdir

def main():
    # Get the month and year of the last closed month (YYYY, MM)
    year_last_closed_month, month_last_closed_month = get_last_closed_month()

    # Set paths for monthly
    monthly_path = './data/spot/monthly/klines'

    # Create directories if they don't exist
    Path(monthly_path).mkdir(parents=True, exist_ok=True)
    Path('./download_tracking').mkdir(parents=True, exist_ok=True)

    # Get list of tickers that already have kline data
    existing_tickers = sorted([f for f in listdir(monthly_path)])

    # Get all trackers that already exist, and extract tickers
    trackers = list(set(glob(r'./download_tracking/*.pkl', recursive=True)))
    trackers_tickers = [t.split('/')[-1][:-4] for t in trackers]

    # Get list of all available trading symbols on Binance
    symbols = get_all_symbols('spot')

    # Check for new symbols that do not have kline data yet
    def check_new_symbols(tickers, symbols):
        return [x for x in symbols if x not in tickers]

    # Check for new symbols that do not have kline data yet
    new_tickers = check_new_symbols(existing_tickers, symbols)
    # new_tickers = new_tickers[:10] #------------------------------------------ONLY 10 SAMPLE TICKERS----------------------------------------------------------->>>> REMOVE THIS EVENTUALLY!!! THIS IS FOR TESTING ONLY

    # Check for symbols that do not have trackers yet
    new_trackers = check_new_symbols(trackers_tickers, symbols)

    # Add new_trackers to new_tickers
    new_tickers = new_tickers + new_trackers

    # If there are new symbols, download kline data for each one
    if len(new_tickers) > 0:
        # Set command line arguments for downloading kline data
        argv = ['-c', '1']
        parser = get_parser('klines')
        args = parser.parse_args(argv)
        # Shuffle list of new symbols to download in a random order
        random.shuffle(new_tickers)
        # Set number of new symbols to download
        num_symbols = len(new_tickers)
        # Get trading type, intervals, years, months, start and end dates, download folder, and checksum flag from command line arguments
        trading_type, symbols, num_symbols, intervals, years, months, start_date, end_date, folder, checksum = args.type, \
                                    new_tickers, num_symbols, args.intervals, args.years, args.months, \
                                    args.startDate, args.endDate, args.folder, args.checksum
        
        # years = ['2023'] #------------------------------------------------ONLY FOR 2023----------------------------------------------------->>>> REMOVE THIS EVENTUALLY!!! THIS IS FOR TESTING ONLY

        # Download kline data and save download tracking information for each new symbol
        for symbol in new_tickers:
            # Create dictionary to hold download tracking information for each interval
            symbol_dict = {}
            interval = '1m'
            # Set web link to None for now
            symbol_dict[interval] = {'web_link':None, 'files':{}, 'links':{'status_checked':False, 'link_dump':False}}
            for year in years:
                for month in months:
                    if int(year) < year_last_closed_month or (int(year) == year_last_closed_month and int(month) <= month_last_closed_month):
                        # Only download if the year and month are not above the last closed month
                        # Get path for downloading kline data
                        path = get_path(trading_type, "klines", "monthly", symbol, interval)
                        # Set file name for downloaded data
                        file_name = "{}-{}-{}-{}.zip".format(symbol.upper(), interval, year, '{:02d}'.format(month))
                        # If web link is None, set it to path
                        if symbol_dict[interval]['web_link'] == None:
                            symbol_dict[interval]['web_link'] = path
                        # Add download tracking information for each file
                        symbol_dict[interval]['files'][file_name] = {'Downloaded':False, 'Checksum':None, 'Link':None, 'Available':None, 'Download_Tried_and_Succeeded':None}
            # Save download tracking information for this symbol
            pickle.dump(symbol_dict, open('./download_tracking/' + str(symbol) + '.pkl',"wb"))

if __name__ == '__main__':
    main()
