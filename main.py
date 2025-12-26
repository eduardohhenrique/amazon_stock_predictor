from src.config import *
from src.data_loader import amzn_load
from src.preprocess import add_mean_column, add_actual_column, flatten_columns, scale_features, scale_target
from src.plots import plot_mean
import pandas as pd

df = amzn_load()
df = add_mean_column(df)

df_for_prediction = add_actual_column(df)
df_for_prediction = flatten_columns(df_for_prediction)

# Testes
print(df_for_prediction.head(2))
print(df_for_prediction.tail())
print(df_for_prediction.shape)
print(df_for_prediction.columns)

# Plot 'Mean'
plot_mean(df_for_prediction)

# Scaling
x, sc_in = scale_features(df_for_prediction)
y, sc_out = scale_target(df_for_prediction)

x.rename(columns = {
  0: 'Close', 
  1: 'High', 
  2: 'Low', 
  3: 'Open', 
  4: 'Volume', 
  5: 'Mean'
  }, 
  inplace = True)

x = pd.DataFrame(x)
x.index = df_for_prediction.index
print(x.head(2))

y.rename(columns = {0: 'Stock Price Next Day'}, inplace = True)
y.index = df_for_prediction.index
print(y.head(2))
