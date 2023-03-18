# Importing all necessary libraries
import pickle
import numpy as np
import glob
import os
from train_funcs import test_batch
from numba.core.errors import NumbaPendingDeprecationWarning

# Remove warnings
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = NumbaPendingDeprecationWarning)

#find the most recent file that has './binance_dataset_*.pkl' in the name in the folder ./
file = max(glob.iglob('./binance_dataset_*.pkl'), key=os.path.getctime)

def main():
    #find the most recent file that has './binance_dataset_*.pkl' in the name in the folder ./
    file = max(glob.iglob('./binance_dataset_*.pkl'), key=os.path.getctime)

    # Loading training data pickle
    data = pickle.load(open(file, 'rb'))
    print('> Training data pickle loaded.')

    # # Print the types of every value in data, if it's an array, print the shape as well
    # for key, value in data.items():
    #     if type(value) == np.ndarray:
    #         print(key, type(value), value.shape)
    #     else:
    #         print(key, type(value))

    # Setting parameters for test batch
    outsample_size = 48 #hours ahead 
    lookback = 7 #horizons of outsample size 
    batch_size = 1024 * 4200
    insample_size = lookback * outsample_size

    # Running the function for test
    x, x_mask, y, y_mask = test_batch(data, insample_size, outsample_size, batch_size)
    print('> Test batch completed.')
    # Check if there are zeros in y_mask
    # print(f'Number of zeros in y_mask: {np.sum(y_mask == 0)}')

    # Remove rows with zeros in y_mask
    rows_with_zeros = np.where(y_mask == 0)[0]

    # Remove rows with zeros in x, x_mask, y, y_mask
    x = np.delete(x, rows_with_zeros, axis = 0)
    x_mask = np.delete(x_mask, rows_with_zeros, axis = 0)
    y = np.delete(y, rows_with_zeros, axis = 0)
    y_mask = np.delete(y_mask, rows_with_zeros, axis = 0)
    print('> y_mask zeros removed.')
    # # Checking size of arrays
    # print('----------------')
    # print('Checking the size of generated arrays')
    # print('x size:', x.shape)
    # print('x_mask size:', x_mask.shape)
    # print('y size:', y.shape)
    # print('y_mask size:', y_mask.shape)

    # # Check if there are zeros in x_mask
    # print(f'Number of zeros in x_mask: {np.sum(x_mask == 0)}')

    # Remove rows with zeros in x_mask
    rows_with_zeros = np.where(x_mask == 0)[0]

    # Remove rows with zeros in x, x_mask, y, y_mask
    x = np.delete(x, rows_with_zeros, axis = 0)
    x_mask = np.delete(x_mask, rows_with_zeros, axis = 0)
    y = np.delete(y, rows_with_zeros, axis = 0)
    y_mask = np.delete(y_mask, rows_with_zeros, axis = 0)
    print('> x_mask zeros removed.')
    # # Checking size of arrays
    # print('----------------')
    # print('Checking the size of generated arrays')
    # print('x size:', x.shape)
    # print('x_mask size:', x_mask.shape)
    # print('y size:', y.shape)
    # print('y_mask size:', y_mask.shape)

    # # Check if there are zeros in y_mask
    # print(f'Number of zeros in y_mask: {np.sum(y_mask == 0)}')

    # # Check if there are zeros in x_mask
    # print(f'Number of zeros in x_mask: {np.sum(x_mask == 0)}')

    # Saving the pickle
    pickle.dump((x, y), open('test_samples_{l}L_test.pkl'.format(l = lookback), 'wb'))
    print('> test_sample pickle saved sucessfully')

if __name__ == '__main__':
    main()