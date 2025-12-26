import pandas as pd

def predict_sarimax(
    results,
    y_test: pd.Series,
    x_test: pd.DataFrame
) -> pd.Series:
  
    x_test = x_test.loc[y_test.index]
    x_test = x_test.dropna()

    steps = len(x_test)
    forecast = results.get_forecast(
        steps=steps,
        exog=x_test
    )

    predictions = forecast.predicted_mean
    predictions.index = x_test.index

    return predictions
