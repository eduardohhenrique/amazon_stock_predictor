import plotly.express as px
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