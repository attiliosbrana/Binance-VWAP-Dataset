# Binance-VWAP-Dataset/download_data

The code in this folder allows users to regularly update their cryptocurrency data from Binance and preprocess it as volume-weighted average prices (VWAPs), which they can then use to sample random portfolios in their applications.

To use the code in this repository, it is recommended that you activate the necessary conda or Python environment and run each file as follows:

- `python 1_fetch_new_symbols.py` downloads symbols from Binance and creates directories for tracking and data.
- `python 2_download_data.py` downloads data into the directories created in step 1.
- `python 3_checksum.py` applies checksums to detect and fix corrupted files. If there are corrupted files, you may need to run `2_download_data.py` again.
- `python 4_get_denominations.py` gets the available denominations for each cryptocurrency, such as `['USDT', 'USD', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']`.

## Folder Structure

- `data/`: contains downloaded kline .zip files representing raw data klines from Binance.
- `download_tracking/`: contains tracking information for downloaded files.

## Files

- `1_fetch_new_symbols.py`: downloads symbols from Binance and creates directories for tracking and data.
- `2_download_data.py`: downloads data into the directories created in step 1.
- `3_checksum.py`: applies checksums to detect and fix corrupted files.
- `4_get_denominations.py`: gets the available denominations for each cryptocurrency.
- `download_kline.py`: downloads kline data from Binance.
- `enums.py`: contains Enums used in the code.
- `utility.py`: contains utility functions used in the code, which are from the official Binance Vision repo.
- `utils.py`: contains customized utility functions used in the code.
