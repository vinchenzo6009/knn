import sys

import numpy as np
import pandas as pd

def dist(x1, x2, y1, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def most_common(lst):
    return max(set(lst), key=lst.count)

def knn(df, k=10, mesh_step=10):

    x_axis = np.linspace(df['x'].min(), df['x'].max(), mesh_step).round(3)
    y_axis = np.linspace(df['y'].min(), df['y'].max(), mesh_step).round(3)
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

def dist_sum(df):
    # Sum of one set of tri-diagonal elements
    d_mat = df.iloc[:, 2:].to_numpy()
    J = len(d_mat)
    total_sum = sum(d_mat[j][j + 1] for j in range(J - 1)) + d_mat[J - 1][J - 2]
    return total_sum

def insert_coordinate(df, i_from, i_to):
    idx = df.index.tolist()
    idx.insert(i_to, idx.pop(i_from))
    df = df.reindex(idx)
    df = df[['x', 'y'] + idx]
    return df

def dist_matrix(x, y):
    df = pd.DataFrame({'x': x, 'y': y})

    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    n = len(x)
    for i in range(0, n):
        df[i] = dist(x1=x,
                     x2=df.iloc[i]['x'],
                     y1=y,
                     y2=df.iloc[i]['y'])

    df.columns = ['x', 'y'] + df.index.tolist()

    return df


def order_coordinates(clusters):

    # Sort coordinates
    ordered = {}
    for c in clusters:
        df = dist_matrix(clusters[c]['x'], clusters[c]['y'])
        ordered[c] = {}
        idx = df.index.tolist()
        n = len(idx)
        df0 = df.loc[idx, ['x', 'y'] + idx]
        df0.reset_index(drop=True, inplace=True)
        df0.columns = ['x', 'y'] + df0.index.tolist()

        updated = True
        while updated:
            updated = False
            for i in range(0, n):
                # Sum of coordinate path
                sum0 = dist_sum(df=df0)
                for j in range(0, n):
                    if (i > 0) and (i != j):
                        df1 = insert_coordinate(df=df0, i_from=j, i_to=i - 1)
                        sum1 = dist_sum(df=df1)
                        if sum1 < sum0:
                            df0 = df1
                            updated = True

                    if (i < n - 1) and (i != j):
                        df1 = insert_coordinate(df=df0, i_from=j, i_to=i + 1)
                        sum1 = dist_sum(df=df1)
                        if sum1 < sum0:
                            df0 = df1
                            updated = True

        ordered[c]['x'] = df0['x'].values
        ordered[c]['y'] = df0['y'].values

    return ordered

def border2(grid):
    remaining = grid.copy()
    cluster = pd.DataFrame()

    x_border = np.array([])
    y_border = np.array([])

    mesh_step_x = grid[['x']].drop_duplicates().diff().values.tolist()[1][0]
    mesh_step_y = grid[['y']].drop_duplicates().diff().values.tolist()[1][0]
    while not remaining.empty:  # runtime: 26 seconds
        point = remaining.iloc[[0]]
        remaining = remaining[1:]
        cluster = pd.concat([cluster, point], axis=0)

        group = point['class'].values[0]

        while not cluster.empty:
            x0, y0, c0 = cluster.iloc[0]

            # If adjacent points are of the same group, add to cluster
            for x_pm, y_pm in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
                adj_point = grid.loc[
                    (grid['x'] == (x0 + mesh_step_x * x_pm)) &
                    (grid['y'] == (y0 + mesh_step_y * y_pm))]

                # Check if adj. point exists (necessary at the edge of the grid)
                if not adj_point.empty:
                    x1, y1, c1 = adj_point.values[0]

                    # If adjacent points are of the same group, and not already added to cluster, add it and remove from remaining
                    if ((c1 == group) &
                            (adj_point.index not in cluster.index.tolist()) &
                            (adj_point.index in remaining.index.tolist())):
                        cluster = pd.concat([cluster, adj_point], axis=0)
                        remaining = remaining.drop(index=adj_point.index)

                    # Add border
                    elif adj_point['class'].values[0] != group:
                        x_border = np.append(x_border, x0 + (x1 - x0) / 2)
                        y_border = np.append(y_border, y0 + (y1 - y0) / 2)

            # Point completely evaluated in cluster and it can now be disregarded
            cluster = cluster[1:]

    edge = order_coordinates(x=x_border, y=y_border, mesh_step=(mesh_step_x, mesh_step_y))

    return edge

def fill_diagonal(df, value=0):
    mat = df.values
    n = mat.shape[0]
    mat[range(n), range(n)] = value
    return pd.DataFrame(mat)

def get_unique(x, y, eps = 0.1):
    df = dist_matrix(x, y).iloc[:, 2:]
    df = fill_diagonal(df, value = df.max().max())

    df = df[df > eps].dropna(how='any')
    idx = df.index.tolist()

    return x[idx], y[idx]

def get_subset(df, x, y):

    res = df[(df['x'] >= min(x)) &
             (df['x'] <= max(x)) &
             (df['y'] >= min(y)) &
             (df['y'] <= max(y))]

    return res

def remove_duplicates(df, mesh_step):
    d_mat = dist_matrix(df['x'], df['y'])
    idx = d_mat.index.tolist()
    idx_unique = idx.copy()

    while idx:
        # Check if dist from coord to other coordinates is appriximately zero
        coord = idx[0]
        idx.remove(coord)
        df_i = d_mat.loc[coord, idx]
        duplicates = df_i[df_i < min(mesh_step[0], mesh_step[1]) * 0.1].index.tolist()

        for d in duplicates:
            idx.remove(d)
            idx_unique.remove(d)

    return df.loc[idx_unique, :].reset_index(drop=True)

def overwrite_coordinates(df_old, df_new, eps):

    n = max(df_new.index) + 1
    df_old.index = range(n, n + df_old.shape[0])
    df_combined = pd.concat([df_new, df_old], axis=0)

    # Dist from df_old to df_new
    dist_mat = dist_matrix(df_combined['x'], df_combined['y'])
    df_s = dist_mat.loc[df_old.index, df_new.index]

    # df_new present in df_old
    df_s = df_s < eps
    idx = df_s.any(axis=0).index

    # Overwrite
    return df_new.loc[idx, :]

def get_cluster_edge_coordinates(grid):
    mesh_step = (np.round(grid['x'].drop_duplicates().diff().values.tolist()[1], 3),
                 np.round(grid['y'].drop_duplicates().diff().values.tolist()[1], 3))

    cluster_edge = {}
    for c in grid['class'].unique():  # c='group1'
        df = grid[grid['class'] == c]
        x = np.array([])
        y = np.array([])
        for x_pm, y_pm in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
            x = np.append(x, (df['x'] + x_pm * 0.5 * mesh_step[0]).to_numpy())
            y = np.append(y, (df['y'] + y_pm * 0.5 * mesh_step[1]).to_numpy())

        # Get unique elements in class c
        t = pd.DataFrame({'x': x, 'y': y})
        x, y = get_unique(x, y, eps=min(mesh_step[0], mesh_step[1]) * 0.9)
        df = pd.DataFrame({'x': x, 'y': y})
        df = get_subset(df=df, x=(grid['x'].min(), grid['x'].max()), y=(grid['y'].min(), grid['y'].max()))
        cluster_edge[c] = df

    return cluster_edge

def get_edge_sections(cluster_edge, grid):
    mesh_step = (np.round(grid['x'].drop_duplicates().diff().values.tolist()[1], 3),
                 np.round(grid['y'].drop_duplicates().diff().values.tolist()[1], 3))
    s = 0
    sections = {s: []}

    # Clustering
    for c in cluster_edge.keys():
        d_mat = dist_matrix(cluster_edge[c]['x'], cluster_edge[c]['y'])
        df = pd.concat([cluster_edge[c]['x'], cluster_edge[c]['y']], axis=1)

        remaining = df.index.tolist()
        done = []
        section = [remaining[0]]

        while remaining:
            remaining_section_coordinates = [point for point in section if point not in done]
            if remaining_section_coordinates:
                i = remaining_section_coordinates[0]
                x = df.loc[i, 'x']
                y = df.loc[i, 'y']

                # Box condition
                adj1 = get_subset(df, (x - mesh_step[0], x + mesh_step[0]),
                                  (y - mesh_step[1], y + mesh_step[1])).index.tolist()

                # Corner condition
                t = d_mat.loc[i, adj1]
                adj2 = t[t <= max(mesh_step[0], mesh_step[1]) * 1.1].index.tolist()

                # Grid point condition
                adj = list(set.intersection(set(adj1), set(adj2)))
                for a in adj:
                    adj_x = df.loc[a, 'x']
                    adj_y = df.loc[a, 'y']
                    # Check if adjacent point is on the same axis
                    if (abs((adj_x - x) / (0.5 * mesh_step[0])) < 0.5 or
                            abs((adj_y - y) / (0.5 * mesh_step[1])) < 0.5):
                        # Check if there is a grid point in between
                        if not get_subset(grid, x=(x, adj_x), y=(y, adj_y)).empty:
                            adj.remove(a)

                section = list(set(section + adj))
                done.append(i)
                remaining.remove(i)
            else:
                sections[s] = df.loc[section, ['x', 'y']]
                section = [remaining[0]]
                s += 1

        sections[s] = df.loc[section, ['x', 'y']]

    # Filter out duplicates
    sections = {k: v for k, v in sections.items() if v.index.tolist() not in [sections[j].index.tolist() for j in range(k)]}

    return sections

def border(grid):
    mesh_step = (np.round(grid['x'].drop_duplicates().diff().values.tolist()[1], 3),
                 np.round(grid['y'].drop_duplicates().diff().values.tolist()[1], 3))

    cluster_edge = get_cluster_edge_coordinates(grid)

    # Identify truly unique coordinates
    e = pd.DataFrame({'x': [], 'y': []})
    for c in cluster_edge.keys():
        temp = cluster_edge[c]
        e = pd.concat([e, temp], axis=0, ignore_index=True).sort_values(['x', 'y']).reset_index(drop=True)
    e = remove_duplicates(e, mesh_step)

    # Align cluster edge coordinates
    for c in cluster_edge.keys():
        df_c = cluster_edge[c]
        cluster_edge[c] = overwrite_coordinates(df_old = df_c, df_new = e, eps = min(mesh_step) * 0.1)

    # Split into separate lines
    sections = get_edge_sections(cluster_edge, grid)
    # Handle connecting edges <-

    edge = order_coordinates(sections)

    return edge

#plt.scatter(cluster_edge['group1']['x'], cluster_edge['group1']['y'], color='r', s=10)
#plt.scatter(cluster_edge['group2']['x'], cluster_edge['group2']['y'], color='g', s=10)
#plt.scatter(edge_coordinates['x'], edge_coordinates['y'], color='b', s=10)
#plt.scatter(df0['x'], df0['y'], color='b', s=10)