from train import eval_weights
from data import training_data
import pandas as pd
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx


bounds = {
    'f0': [6.5, 7, 40],
    'w0': [3.8, 4.7, 40],
    'beta': [0.2, 0.8, 100]
}
diffs = [(b[1] - b[0]) / b[2] for b in bounds.values()]


def brute_force(run, train_data, bounds, sorted=False):
    results = pd.DataFrame(columns=['f0', 'w0', 'beta', 'err'])
    bounds = {k: np.arange(v[0], v[1], (v[1]-v[0])/v[2]) for k, v in bounds.items()}
    train_data = np.array(train_data)
    train_data = [train_data.take(0, 1), train_data.take(1, 1)]

    print('Trying weight combinations...')
    for weights in itertools.product(*bounds.values()):
        err = eval_weights(dict(zip(bounds.keys(), weights)), train_data, nparr=True)
        results.loc[len(results)] = [*weights, err]

    results.to_csv(f'data/mapping/brute_force_{run}.csv')
    if sorted:
        results = dict(sorted(results.items(), key=lambda x: x[1]))
    return results

def calculate_diff_map(run):
    df = pd.read_csv(f'data/mapping/brute_force_{run}.csv')

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

    out.to_csv(f'data/mapping/diff_map_{run}.csv')


run = 3
compute_map = True

if compute_map:
    brute_force(run, training_data(run), bounds)
    calculate_diff_map(run)

display = 'd_dbeta'
sampling_slice = 3
df = pd.read_csv(f'data/mapping/diff_map_{run}.csv')

# slice up the data
scale = 1
df = df[::sampling_slice] / scale
# df = df[df['beta'] <= 0.12]
# df = df[df['beta'] <= 0.3]
# df = df[df['w0'] > 4.2]
df = df[df[display] <= 2]
df = df[df[display] >= -2]

# generate plot
cm = plt.get_cmap('jet')
cNorm = matplotlib.colors.Normalize(vmin=min(df[display]), vmax=max(df[display]))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')
ax3d.scatter(df['f0'], df['w0'], df['beta'], c=scalarMap.to_rgba(df[display]))

ax3d.set_xlabel('$f_0$', fontsize=20)
ax3d.set_ylabel('$\\omega_0$', fontsize=20)
ax3d.set_zlabel('$\\beta$', fontsize=20)

ax3d.xaxis.set_rotate_label(False)
ax3d.yaxis.set_rotate_label(False)
ax3d.zaxis.set_rotate_label(False)

scalarMap.set_array(df[display])
fig.colorbar(scalarMap)

plt.title(f'Run {run} - ' + r'$\frac{\partial}{\partial \beta}<Error^2>$')
plt.show()
