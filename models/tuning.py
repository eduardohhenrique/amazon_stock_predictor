from pmdarima import auto_arima

def tuning_model(
  y_train,
  x_train = None,
  seasonal = False,
):


  model = auto_arima(
    y_train,
    exogenous = x_train,
    seasonal = seasonal,
    #m = m,

    start_p = 0, start_q = 0,
    max_p = 2, max_q = 2,

    start_P = 0, start_Q = 0,
    max_P = 1, max_Q = 1,

    d = None,
    D = None,

    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    n_jobs = 1
    )

  return model
