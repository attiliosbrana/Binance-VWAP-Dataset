# Binance-VWAP-Dataset/benchmarking

This folder aims to provide a benchmarking framework for evaluating new methods against the test set used in the paper N-BEATS Perceiver [1]. 

The Jupyter notebook `benchmark_example.ipynb` allows users to test their own models using the x and y data from the test files and compare their results with those obtained in the original N-BEATS Perceiver paper [1]. The notebook contains code that downloads and saves the test files from the Open Science Framework, reassembles the original arrays from the splits, and separates the x and y data into training and testing sets. Then, a LinearRegression model is developed and trained using the training data as an example of how users could apply their own models into this benchmarking framework. This model is subsequently used to predict y values for the test data. Metrics such as MSE, MAE, and SMAPE are calculated for the predictions, with average metrics being computed and displayed. Users can compare their model's performance against the performance of the models presented in [1].

Additionally, the `metrics_fast.py` file offers a Python module containing functions for calculating various metrics used in time series analysis. These functions enable highly efficient computation of these metrics for numerous series, utilizing vectorized calculations in numpy. The included metrics are:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Symmetric Mean Absolute Percentage Error (SMAPE)
- Mean Absolute Scaled Error (MASE)
- Root Mean Squared Scaled Error (RMSSE)
- R-Squared Score (R2)

Each function accepts two or more numpy arrays containing the true and predicted values and returns a scalar value representing the corresponding metric.

The `RMSSE()` function requires four arguments: `y_true`, `y_pred`, `y_train`, and `sp`. `y_true` and `y_pred` are numpy arrays containing the true and predicted values, respectively. `y_train` is a numpy array containing the training data, while `sp` is an integer denoting the seasonal period. This function returns a scalar value representing the root mean squared scaled error.

The `r2_score()` function takes two arguments: `y` and `y_hat`, which are numpy arrays containing the true and predicted values, respectively. It returns a numpy array of scalar values representing the R-squared score for each time series in `y` and `y_hat`.

The `calculate_metrics()` function requires four arguments: `y`, `y_hat`, `x`, and `sp`. `y` and `y_hat` are numpy arrays containing the true and predicted values, respectively. `x` is a numpy array containing the training data, and `sp` is an integer denoting the seasonal period. This function calls the aforementioned functions to calculate all metrics, storing the results in a Pandas DataFrame with the following columns: 'mase', 'rmsse', 'mae', 'rmse', 'mape', 'smape', 'r2'. It returns the DataFrame.

References:
[1] Attilio Sbrana, Paulo André Lima de Castro. N-BEATS Perceiver: A Novel Approach for Robust Cryptocurrency Portfolio Forecasting, 23 February 2023, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-2618277/v1]
