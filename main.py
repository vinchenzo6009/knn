import pandas as pd
from graphs import *
from functions import knn

data = pd.read_csv('data.csv', header=0)

grid = knn(df=data, mesh_step=50)

plot_k_means(df=data, grid=grid)

plt.show()
