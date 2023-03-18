import numpy as np
import pandas as pd

# Define functions for calculating error metrics

def mean_absolute_error(y, y_hat):
    """
    Calculate the mean absolute error (MAE) between two arrays.
    
    Parameters:
    y (numpy array): Ground truth array.
    y_hat (numpy array): Predicted array.
    
    Returns:
    float: Mean absolute error between y and y_hat.
    """
    return np.mean(np.abs(y - y_hat), axis=-1)

def mean_squared_error(y, y_hat):
    """
    Calculate the mean squared error (MSE) between two arrays.
    
    Parameters:
    y (numpy array): Ground truth array.
    y_hat (numpy array): Predicted array.
    
    Returns:
    float: Mean squared error between y and y_hat.
    """
    return np.mean((y - y_hat)**2, axis=-1)

def root_mean_squared_error(y, y_hat):
    """
    Calculate the root mean squared error (RMSE) between two arrays.
    
    Parameters:
    y (numpy array): Ground truth array.
    y_hat (numpy array): Predicted array.
    
    Returns:
    float: Root mean squared error between y and y_hat.
    """
    return np.sqrt(mean_squared_error(y, y_hat))

def mean_absolute_percentage_error(y, y_hat):
    """
    Calculate the mean absolute percentage error (MAPE) between two arrays.
    
    Parameters:
    y (numpy array): Ground truth array.
    y_hat (numpy array): Predicted array.
    
    Returns:
    float: Mean absolute percentage error between y and y_hat.
    """
    return 100 * np.mean(np.abs((y - y_hat) / y), axis=-1)

def symmetric_mean_absolute_percentage_error(forecast, target):
    """
    Calculate the symmetric mean absolute percentage error (SMAPE) between two arrays.
    
    Parameters:
    forecast (numpy array): Predicted array.
    target (numpy array): Ground truth array.
    
    Returns:
    float: Symmetric mean absolute percentage error between forecast and target.
    """
    return 200 * np.mean(np.divide(np.abs(forecast - target),
                                      np.abs(forecast) + np.abs(target)), axis=-1)

def mean_absolute_scaled_error(insample, freq, forecast, target):
    """
    Calculate the mean absolute scaled error (MASE) between two arrays.
    
    Parameters:
    insample (numpy array): In-sample array used for scaling.
    freq (int): Frequency of the data (e.g. 7 for weekly, 12 for monthly).
    forecast (numpy array): Predicted array.
    target (numpy array): Ground truth array.
    
    Returns:
    float: Mean absolute scaled error between forecast and target.
    """
    masep = np.mean(np.abs(insample[:, freq:] - insample[:, :-freq]), axis=1)
    return np.mean(np.abs(target - forecast) / masep[:, None], axis=-1)


def RMSSE(y_true, y_pred, y_train, sp = 1):
    """
    Calculate the Root Mean Squared Scaled Error (RMSSE).

    Parameters:
    y_true (array-like): Actual (true) values.
    y_pred (array-like): Predicted values.
    y_train (array-like): Actual values of training data.
    sp (int): Seasonality period.

    Returns:
    float: RMSSE value.
    """
    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Calculate the mean squared error of the training data
    mse_naive = mean_squared_error(y_train[:, sp:], y_train[:, :-sp])

    # Calculate the scaling factor
    scaling_factor = np.sqrt(mse_naive)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(mse)

    # Calculate the RMSSE
    rmsse = rmse / scaling_factor

    return rmsse


def r2_score(y, y_hat):
    """
    Calculate the R^2 (coefficient of determination) regression score function.

    Parameters:
    y (array-like): Actual (true) values.
    y_hat (array-like): Predicted values.

    Returns:
    float: R^2 score value.
    """
    y_mean = np.mean(y, axis = 1).reshape(-1, 1)
    return 1 - np.sum((y - y_hat)**2, axis = 1) / np.sum((y - y_mean)**2, axis = 1)


def calculate_metrics(y, y_hat, x, sp=1):
    """
    Calculate all the evaluation metrics and return the result in a DataFrame.

    Parameters:
    y (array-like): Actual (true) values.
    y_hat (array-like): Predicted values.
    x (array-like): Training data.
    sp (int): Seasonality period.

    Returns:
    pandas.DataFrame: A DataFrame containing all the evaluation metrics.
    """
    metrics = {}
    metrics['mase'] = mean_absolute_scaled_error(x, sp, y_hat, y)
    metrics['rmsse'] = RMSSE(y, y_hat, x, 1)
    metrics['mae'] = mean_absolute_error(y, y_hat)
    metrics['rmse'] = root_mean_squared_error(y, y_hat)
    metrics['mape'] = mean_absolute_percentage_error(y, y_hat)
    metrics['smape'] = symmetric_mean_absolute_percentage_error(y, y_hat)
    metrics['r2'] = r2_score(y, y_hat)
    #now make it a dataframe
    metrics = pd.DataFrame(metrics)
    return metrics