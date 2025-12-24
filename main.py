from src.data_loader import amzn_load

df = amzn_load()
print(df.head(2))
print(df.tail())
print(df.shape)