import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

def plot_mean(df):
  fig = px.line(
    df,
    x = df.index,
    y = 'Mean',
    labels = {'x': 'Date'},
    title = 'Time Series - Amazon (AMZN)'
  )
  
  fig.update_xaxes(rangeslider_visible = True)
  
  plot_mean_name = 'plot_mean'
  
  # Folders
  img_dir = Path('reports/figures')
  html_dir = Path('reports/interactive')
  
  img_dir.mkdir(parents = True, exist_ok = True)
  html_dir.mkdir(parents = True, exist_ok = True)
  
  # Save
  fig.write_image(img_dir/ f'{plot_mean_name}.png')
  fig.write_html(html_dir/ f'{plot_mean_name}.html')
  
  print(f"Graphic PNG saved in '{img_dir}'.")
  print(f"And the HTML saved in '{html_dir}'.")
  
def plot_seas_mean(df, column = 'Mean', period = 252):
  seas_dec = sm.tsa.seasonal_decompose(
    df[column],
    model = 'add',
    period = period
    )
  
  seas_desc_name = 'seas_dec'
  
  plt.rcParams['figure.figsize'] = (15, 4)
  
  fig = seas_dec.plot()
  fig.set_figheight(4)
  
  # Adjust color
  for ax in fig.axes:
    ax.lines[0].set_color('#1f77b4')
    
  # Folders
  img_dir = Path('reports/figures')
  img_dir.mkdir(parents = True, exist_ok = True)
  
  # Save
  plt.savefig(img_dir/ f'{seas_desc_name}.png', bbox_inches = 'tight')
  plt.close()

  print(f"Graphic PNG saved in '{img_dir}'.")


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_actual_vs_predicted(
    y_test: pd.Series,
    predictions: pd.Series,
    title: str = "Actual vs Predicted"
):
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.plot(y_test.index, y_test, label="Actual", color="blue")
    ax.plot(predictions.index, predictions, label="Predicted", color="red")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    # Pasta de saída
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "actual_vs_predicted.png", bbox_inches="tight")
    plt.close()
