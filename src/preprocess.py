import pandas as pd  

# Add the average between 'Low' and 'High' column
def add_mean_column(df):
  df = df.copy()
  df['Mean'] = (df['Low'] + df['High']) / 2
  
  return df


# Add the 'Actual' price column
def add_actual_column(df):
  steps = -1
  
  df_for_prediction = df.copy()
  df_for_prediction['Actual'] = df_for_prediction['Mean'].shift(steps)
  df_for_prediction = df_for_prediction.dropna()
  
  return df_for_prediction

# Remove MultiIndex
def flatten_columns(df):
  df = df.copy()
  
  df.columns = [
    '_'.join(col).strip() if isinstance(col, tuple) else col 
    for col in df.columns
  ]

# Rename columns
  df = df.rename(columns={
    "Close_AMZN": "Close",
    "High_AMZN": "High",
    "Low_AMZN": "Low",
    "Open_AMZN": "Open",
    "Volume_AMZN": "Volume",
    "Mean_": "Mean",
    "Actual_": "Actual"
    })
  
  return(df)