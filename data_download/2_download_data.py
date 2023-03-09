from glob import glob
import pickle
from pathlib import Path

# Binance API Python Official Utilities
from utility import download_file

def main():
    # Get all trackers that already exist, and extract tickers
    trackers = list(set(glob(r'./download_tracking/*.pkl', recursive=True)))
    tickers = [t.split('/')[-1][:-4] for t in trackers]

    # Download kline data for each symbol
    for tracker, symbol in zip(trackers, tickers):
        # Load download tracking information for the current symbol
        symbol_dict = pickle.load(open(tracker, "rb"))
        interval = '1m'
        path = symbol_dict[interval]['web_link']
        
        # Download each file and its corresponding checksum if it hasn't been downloaded yet
        for file in symbol_dict[interval]['files']:
            file_dict = symbol_dict[interval]['files'][file]
            if 'Download_Tried_and_Succeeded' in file_dict.keys():
                if file_dict['Downloaded'] == False and file_dict['Download_Tried_and_Succeeded'] == None:
                    download_file(path, file, None, None)
                    download_file(path, file + '.CHECKSUM', None, None)
                    # Check if the download was successful, by checking if the file exists
                    if Path(path + file).is_file() and Path(path + file + '.CHECKSUM').is_file():
                        file_dict['Download_Tried_and_Succeeded'] = True
                #Check if it's a checksum redownload
                elif file_dict['Downloaded'] == True and file_dict['Checksum'] == False:
                    download_file(path, file, None, None)
                    download_file(path, file + '.CHECKSUM', None, None)
                    # Check if the download was successful, by checking if the file exists
                    if Path(path + file).is_file() and Path(path + file + '.CHECKSUM').is_file():
                        file_dict['Checksum'] = None 
        
        # Save updated download tracking information for the current symbol
        pickle.dump(symbol_dict, open('./download_tracking/' + str(symbol) + '.pkl', "wb"))

if __name__ == '__main__':
    main()