import numpy as np
import pandas as pd

def dist(x1, x2, y1, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def most_common(lst):
    return max(set(lst), key=lst.count)

def k_means(df, k=10, mesh_step=0.5):
    x_axis = np.arange(df['x'].min(), df['x'].max(), step=mesh_step)
    y_axis = np.arange(df['y'].min(), df['y'].max(), step=mesh_step)
    xv, yv = np.meshgrid(x_axis, y_axis)

    grid = pd.DataFrame({'x': xv.flatten(),
                         'y': yv.flatten(),
                         'class': ['unknown'] * len(xv.flatten())})

    for x, y, i in zip(grid['x'], grid['y'], range(0, len(grid))):
        distance_to_grid = dist(df['x'], x, df['y'], y)
        temp = pd.concat([df, pd.DataFrame(distance_to_grid, columns=['distanceToGrid'])], axis=1)
        k_nearest = temp.sort_values('distanceToGrid', ascending=True)['class'][:k].values.tolist()
        grid.loc[i, 'class'] = most_common(k_nearest)

    return grid
