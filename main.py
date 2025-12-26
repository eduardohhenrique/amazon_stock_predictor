from src.config import *
from src.data_loader import amzn_load
from src.preprocess import add_mean_column, add_actual_column, flatten_columns, scale_features, scale_target
from src.plots import plot_mean, plot_seas_mean
from src.eda import test_adf
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

print(x.head(2))
print(y.head(2))

# Plot Seasonal Decompose of 'Mean'
plot_seas_mean(df_for_prediction)

# Train / Test split
train_size = int(len(df) * 0.80)
test_size = len(df) - train_size

x_train = x[:train_size].dropna()
x_test = x[train_size:].dropna()

y_train = y[:train_size].dropna()
y_test = y[train_size:].dropna()
y_test = y['Stock Price Next Day'][train_size:].dropna()

# Test Stationary
test_adf(y_test.diff(), 'Stock Price Next Day')