import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime

denoms = ['AUD', 'BNB', 'BRL', 'BTC', 'ETH', 'EUR', 'RUB', 'USD', 'TRY', 'USDT', 'USDC', 'BIDR', 'IDR', 'GYEN']

cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

# Function to check the SHA256 checksum of a file
def check_the_sum(file):
    # Get path for checksum file
    checksum_file = file + '.CHECKSUM'
    # Calculate SHA256 checksum for the file
    with open(file, "rb") as f:
        bytes = f.read() # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest()
    # Check if there is a checksum file available
    path = Path(checksum_file)
    if path.is_file():
        # If there is a checksum file, compare the checksum with the expected value
        expected_checksum = path.read_text().split(' ')[0]
        return readable_hash == expected_checksum, readable_hash, expected_checksum, checksum_file
    else:
        # If there is no checksum file, return that fact
        return False, 'no checksum'

# Get the month and year of the last closed month
def get_last_closed_month():
    # Get current date and time
    now = datetime.now()
    # Get current year and month
    year = now.year
    month = now.month
    # If current month is January, set last month to December of previous year
    if month == 1:
        year = year - 1
        month = 12
    # If current month is not January, set last month to previous month
    else:
        month = month - 1
    # Return last month and year
    return year, month