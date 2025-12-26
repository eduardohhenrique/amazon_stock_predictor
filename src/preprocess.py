import pandas as pd  
from sklearn.preprocessing import MinMaxScaler

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


# MinMaxScaler

def scale_features(df):
  features = [
    'Close', 
    'High', 
    'Low', 
    'Open', 
    'Volume', 
    'Mean', 
    'Actual'
  ]
  
  sc_in = MinMaxScaler(feature_range = (0, 1))
  scaled_x = sc_in.fit_transform(df[features])
  
  x = pd.DataFrame(
    scaled_x,
    columns = features,
    index = df.index
  )
  
  return x, sc_in


def scale_target(df):
  sc_out = MinMaxScaler(feature_range = (0, 1))
  sclaed_y = sc_out.fit_transform(df[['Actual']])
  
  y = pd.DataFrame(
    sclaed_y,
    columns = ['Actual'],
    index = df.index
  )
  
  return y, sc_out
