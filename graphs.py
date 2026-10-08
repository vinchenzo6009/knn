import matplotlib.pyplot as plt

def plot_scatter(df):
    for group in df['class'].unique():
        subset = df[df['class'] == group]
        plt.scatter(subset['x'], subset['y'], label=group)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_k_means(df, grid):
    colors = {
        'group1': 'C0',
        'group2': 'C1'
    }
    for group in grid['class'].unique():
        subset = grid[grid['class'] == group]
        plt.scatter(subset['x'],
                    subset['y'],
                    s=1,
                    color=colors[group])

    for group in df['class'].unique():
        subset = df[df['class'] == group]
        plt.scatter(subset['x'],
                    subset['y'],
                    marker='x',
                    color=colors[group],
                    label=group)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()
