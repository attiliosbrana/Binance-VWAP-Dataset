# Binance-VWAP-Dataset/pre-processing

This folder contains code that allows users to preprocess raw data from Binance into volume-weighted average prices (VWAPs), which can be used for various applications.

To use the code in this repository, activate the appropriate conda or Python environment and run each file as follows:

`1_preprocess.py` calculates the VWAP for various cryptocurrencies obtained from Binance. It includes base assets like USDT, USD, USDC, BUSD, BTC, ETH, and BNB, as well as converted assets such as BTC, ETH, and BNB. The script saves the VWAP data for each asset in a Python dictionary as a pickle file.

`2_gen_training_data.py` generates training data for cryptocurrency portfolio forecasting. It loads VWAP data for various cryptocurrencies from a pickle file and creates a dictionary for training data using a train_cutoff date. Maitaining the train_cutoff date as '2022-07-01 01:00:00' replicates the exact dataset employed in the paper N-BEATS Perceiver [1]. This dataset can then be used to train models with data from before the cutoff date. The script ultimately saves the generated training data as a pickle file named `binance_dataset_yyyymmdd.pkl`, where "yyyymmdd" represents the current date.

`3_gen_train_test_samples.py` generates test samples for a machine learning model.  It loads the most recent training data pickle file, executes the `test_batch()` function to generate test samples, removes rows with zeros in x_mask and y_mask arrays, and saves the final test samples as a pickle file. The `test_batch()` function accepts several parameters, including `insample_size`, `outsample_size`, and `batch_size`. The generated test samples are saved in a pickle file named `test_samples_{l}L_test.pkl`, where `l` denotes the lookback parameter used in the `test_batch()` function. This replicates the exact steps that generated the test dataset in the paper N-BEATS Perceiver [1]. The complete dataset is available for download at OSF (https://osf.io/fjsuh/) or in the `/benchmarking` folder of this repository.

References:
[1] Attilio Sbrana, Paulo André Lima de Castro. N-BEATS Perceiver: A Novel Approach for Robust Cryptocurrency Portfolio Forecasting, 23 February 2023, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-2618277/v1]