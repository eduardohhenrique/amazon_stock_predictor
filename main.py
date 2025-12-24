from src.data_loader import amzn_load
from src.preprocess import add_mean_column, add_actual_column

df = amzn_load()

df = add_mean_column(df)
df = add_actual_column(df)

# Testes
print(df.head(2))
print(df.tail())
print(df.shape)
print(df.columns)