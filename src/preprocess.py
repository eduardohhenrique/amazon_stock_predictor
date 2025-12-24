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