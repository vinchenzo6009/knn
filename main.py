import pandas as pd
from graphs import *
from functions import k_means

data = pd.read_csv('data.csv', header=0)
plot_scatter(data)

grid = k_means(df=data)
plot_k_means(data ,grid)
