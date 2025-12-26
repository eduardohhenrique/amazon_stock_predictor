from src.config import *
from src.data_loader import amzn_load
from src.preprocess import add_mean_column, add_actual_column, flatten_columns, scale_features, scale_target
from src.plots import plot_mean, plot_seas_mean, plot_actual_vs_predicted
from src.eda import test_adf
from models.tuning import tuning_model
from models.model import sarimax_model
from models.predict import predict_sarimax
from statsmodels.tools.eval_measures import rmse
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
x = df_for_prediction[['Close', 'High', 'Low', 'Open', 'Volume', 'Mean']]
y = df_for_prediction['Actual']

print(x.head(2))
print(y.head(2))

# Plot Seasonal Decompose of 'Mean'
plot_seas_mean(df_for_prediction)

# Train / Test split
train_size = int(len(df_for_prediction) * 0.80)
test_size = len(df_for_prediction) - train_size

x_train = x.iloc[:train_size].dropna()
x_test = x.iloc[train_size:].dropna()

y_train = y.iloc[:train_size].dropna()
y_test = y.iloc[train_size:].dropna()

# Test Stationary
test_adf(y_test.diff(), 'Stock Price Next Day')

# Tuning
auto_model = tuning_model(
  y_train = y_train,
  x_train = x_train,
  seasonal = False,
)

print(auto_model.summary())

# Train Model
model = sarimax_model(
  y_train = y_train,
  x_train = x_train,
  order = (0, 1, 1),
  seasonal_order = (0, 0, 0, 0)
)

# Predict
predictions = predict_sarimax(
  results = model,
  y_test = y_test,
  x_test = x_test,
)

print(predictions.head())

# Compare
comparision = pd.DataFrame({
  'Actual': y_test,
  'Predicted': predictions
})

print(comparision.head())

# Plot 'Actual' vs 'Predicted'
plot_actual_vs_predicted(
    y_test = y_test,
    predictions = predictions,
    title = 'SARIMAX – Actual vs Predicted'
)

# Analysing Error
error = rmse(comparision['Actual'], comparision['Predicted'])
print(f'The error was: {error:.3f}.')