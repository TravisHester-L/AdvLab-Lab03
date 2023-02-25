from train import eval_weights
from data import training_data
import pandas as pd
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx


def brute_force(train_data, bounds, sorted=False):
    results = pd.DataFrame(columns=['f0', 'w0', 'beta', 'err'])
    bounds = {k: np.arange(v[0], v[1], (v[1]-v[0])/v[2]) for k, v in bounds.items()}
    for weights in itertools.product(*bounds.values()):
        err = eval_weights(dict(zip(bounds.keys(), weights)), train_data)
        results.loc[len(results)] = [*weights, err]
    results.to_csv('data/mapping/brute_force.csv')
    if sorted:
        results = dict(sorted(results.items(), key=lambda x: x[1]))
    return results

run = 1
train_data = training_data(run)

bounds = {
    'f0': [6, 7, 40],
    'w0': [4, 5, 40],
    'beta': [0, 0.5, 100]
}
diffs = [(b[1] - b[0]) / b[2] for b in bounds.values()]

# test = brute_force(train_data, bounds)
# i = 1
# for k, v in test.items():
#     print(f'({k[0]:.03f}, {k[1]:.03f}, {k[2]:.03f}): {v:.05f}')
#     i += 1
#     if i > 10:
#         break

def calculate_diff_map():
    df = pd.read_csv('data/mapping/brute_force.csv')

    print('Building 3D matrix...')
    size = (bounds['f0'][2], bounds['w0'][2], bounds['beta'][2])
    matrix = np.ndarray(shape=size, dtype=float, order='C')
    i, j, k = 0, 0, 0
    for index, row in df.iterrows():
        k = index % bounds['beta'][2]
        if k == 0 and index != 0:
            j += 1
            if j % bounds['w0'][2] == 0:
                j = 0
                i += 1
        matrix[i][j][k] = row['err']

    print('Calculating differentials...')
    differentials = [
        np.ndarray(shape=size, dtype=float, order='C')
        for i in range(3)
    ]
    for ax in range(3):
        middle = np.apply_along_axis(lambda m: np.convolve(m, (0.5, 0, -0.5), 'valid'), axis=ax, arr=matrix)
        front_edge = np.apply_along_axis(lambda m: [m[1] - m[0]], axis=ax, arr=matrix)
        back_edge = np.apply_along_axis(lambda m: [m[-1] - m[-2]], axis=ax, arr=matrix)
        full = np.append(front_edge, middle, axis=ax)
        full = np.append(full, back_edge, axis=ax)
        differentials[ax] = full / diffs[ax]

    # print('Calculating divergences...')
    # divergence = np.sum(differentials, axis=0)

    print('Writing data to disk...')
    out = pd.DataFrame(columns=['f0', 'w0', 'beta', 'error', 'd_df0', 'd_dw0', 'd_dbeta'])
    out['f0'] = df['f0']
    out['w0'] = df['w0']
    out['beta'] = df['beta']
    out['error'] = df['err']
    out['d_df0'] = pd.Series(np.reshape(differentials[0], (-1,)))
    out['d_dw0'] = pd.Series(np.reshape(differentials[1], (-1,)))
    out['d_dbeta'] = pd.Series(np.reshape(differentials[2], (-1,)))

    out.to_csv('data/mapping/diff_map.csv')


calculate_diff_map()

display = 'd_dw0'
df = pd.read_csv('data/mapping/diff_map.csv')
df = df[::15]
cm = plt.get_cmap('jet')
cNorm = matplotlib.colors.Normalize(vmin=min(df[display]), vmax=max(df[display]))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.scatter(df['f0'], df['w0'], df['beta'], c=scalarMap.to_rgba(df[display]))
scalarMap.set_array(df[display])
fig.colorbar(scalarMap)
plt.show()
