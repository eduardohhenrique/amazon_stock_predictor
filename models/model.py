from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax_model(
  y_train,
  x_train,
  order = (0, 1, 1),
  seasonal_order = (0, 0, 0, 0)
):

  model = SARIMAX(
    y_train,
    exog = x_train,
    order = order,
    enforce_invertibility = False,
    enforce_stationarity = False
)
  
  results = model.fit(method = 'powell', disp = False)
  
  return results