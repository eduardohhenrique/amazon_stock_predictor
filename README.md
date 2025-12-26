Stock Price Prediction Project (Amazon)
Machine Learning / Time Series project to predict Amazon (AMZN) stock prices using the SARIMAX model with exogenous variables.


Model Used
SARIMAX (statsmodels)
Auto ARIMA (pmdarima) for parameter tuning


Project Structure
amazon_stock_predictor/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── plots.py
│   ├── statistics.py
│   ├── models.py
│   ├── predict.py
│   └── metrics.py
│
├── reports/
│   ├── figures/
│   └── interactive/
│
├── main.py
├── requirements.txt
└── README.md


How to Run
pip install -r requirements.txt
python main.py


Notes
SARIMAX uses non-scaled data
Scaling is applied only for feature preparation
Project developed for learning and portfolio purposes