import pandas as pd
import numpy as np

n1 = 30
n2 = 40
n3 = 10
group1 = {
    'x': np.random.normal(3, 2, n1),
    'y': np.random.normal(6, 4, n1),
    'class': ['group1'] * n1
}
group2 = {
    'x': np.random.normal(5, 3, n2),
    'y': np.random.normal(4, 3, n2),
    'class': ['group2'] * n2
}
group3 = {
    'x': np.random.normal(8, 2, n3),
    'y': np.random.normal(2, 2, n3),
    'class': ['group1'] * n3
}

data = pd.concat([pd.DataFrame(group1), pd.DataFrame(group2), pd.DataFrame(group3)])
data.to_csv('data.csv', header=True, index=False)
