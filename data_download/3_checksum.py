from glob import glob
import os
import pickle

# Our utilities
from utils import check_the_sum

def main():
    # Set path for kline data files and download tracking files
    kline_files = list(set(glob(r'./data/spot/monthly/klines/**/**/*.zip', recursive=True)))
    download_trackers = list(set(glob(r'./download_tracking/*.pkl', recursive=True)))


    # Loop over all kline data files to check their checksums and delete any files that do not have a valid checksum
    fails = []
    for file in kline_files:
        # Get symbol, interval, and file name from file path
        split = file.split('/')
        symbol, interval, file_name = split[5], split[6], split[7]
        # Check if there is a download tracking file for this symbol
        if './download_tracking/{}.pkl'.format(symbol) in download_trackers:
            # If there is a download tracking file, check the checksum and update the tracking information
            answer = check_the_sum(file)
            if answer[0] == True:
                # If the checksum matches, mark the file as downloaded and having a valid checksum
                symbol_dict = pickle.load(open('./download_tracking/' + str(symbol) + '.pkl', "rb"))
                track = symbol_dict[interval]['files'][file_name]
                track['Downloaded'], track['Checksum'], track['Available'] = True, True, True
                pickle.dump(symbol_dict, open('./download_tracking/' + str(symbol) + '.pkl', "wb"))
            else:
                # If the checksum does not match, mark the file as downloaded but with an invalid checksum
                symbol_dict = pickle.load(open('./download_tracking/' + str(symbol) + '.pkl', "rb"))
                track = symbol_dict[interval]['files'][file_name]
                track['Downloaded'], track['Checksum'], track['Available'] = True, False, True
                pickle.dump(symbol_dict, open('./download_tracking/' + str(symbol) + '.pkl', "wb"))
                fails.append((file, answer))
                # Delete the file
                os.remove(file)
                os.remove(file + '.CHECKSUM')
                #print the details of the failure
                print('File: {}\n\tChecksum: {}\n\tExpected: {}\n\tReason: {}'.format(file, answer[1], answer[2], answer[3]))
        else:
            # If there is no download tracking file, add the symbol to the fails list 
            fails.append((file, ('no tracking file',)))

    #report failures
    print(fails)

    #print all fails that are not 'no tracking file'
    for fail in fails:
        if fail[1][0] != 'no tracking file':
            print(fail)

if __name__ == '__main__':
    main()