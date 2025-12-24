from src.config import *
from src.data_loader import amzn_load
from src.preprocess import add_mean_column, add_actual_column, flatten_columns
from src.plots import plot_mean

df = amzn_load()

df = add_mean_column(df)
df = add_actual_column(df)
df = flatten_columns(df)

plot_mean(df)

# Testes
print(df.head(2))
print(df.tail())
print(df.shape)
print(df.columns)

# Plot 'Mean'
plot_mean(df)