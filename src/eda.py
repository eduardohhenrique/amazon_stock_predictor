import statsmodels as sm
from statsmodels.tsa.stattools import adfuller

def test_adf(series, title = ''):
  dfout = {}
  
  dftest = adfuller(
    series.dropna(),
    autolag = 'AIC',
    regression = 'ct'
    )
  
  for key, val in dftest[4].items():
    dfout[f'Critical Value ({key})'] = val
    
  if dftest[1] <= 0.05:
    print("Stationary data! - ", title)
  else:
    print("Non-stationary data! - ", title)