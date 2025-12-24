import pandas as pd
import yfinance as yf

# Downloading the dataset with Yfinance api
def amzn_load(start = '2015-01-01', end = '2025-01-01'):
  df = yf.download(
    'AMZN',
    start = start,
    end = end,
  )
  
  if df.empty:
    raise ValueError('No data for AMZN')

  return df
