import re
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt

rundir_pattern = re.compile(r'(\w+)-n(\d+)-(\w+)(?:-T([\d.]+))?')
datasets = ['4squares', 'checkerboard', 'pinwheel', 'swissroll']
datasets_labels = ['4 squares', 'checkerboard', 'pinwheel', 'Swiss roll']
modes = ['target', 'follmer', 'trig', 'vesde', 'vae', 'gan', 'nnkcde', 'flexcode']
labels = ['Target', 'Föllmer', 'Trigonometric', 'VE-SDE', 'VAE', 'WGAN', 'NNKCDE', 'FlexCode']

table = {}
for rundir in os.listdir('logs'):
    match = rundir_pattern.search(rundir)
    if not match:
        print(f'{rundir} does not match rundir_pattern')
        continue
    dataset, n, mode, T = match.groups()
    T = float(T) if T else None
    n = int(n)
    if (mode == 'follmer' and T is not None or n != 50000) or dataset not in datasets:
        continue
    data = np.load(os.path.join('logs', rundir, 'samples.npz'))
    table.setdefault(dataset, {})
    table[dataset][mode] = data

# load references
for dataset in datasets:
     with open(f'dataset/{dataset}.pkl', 'rb') as f:
        data = pickle.load(f)
        x, y = data['x_train'][:5000], data['y_train'][:5000]
        table[dataset]['target'] = dict(x=x, y=y)

fig, axes = plt.subplots(len(datasets), len(modes), figsize=(10, 6), gridspec_kw={'wspace': 0.2, 'hspace': 0.2}, layout=None)
for j, mode in enumerate(modes):
    for i, dataset in enumerate(datasets):
        ax = axes[i][j]
        data = table[dataset][mode]
        x, y = data['x'], data['y']
        ax.scatter(x, y, s=0.5, marker='o', facecolors='C0', edgecolors='none', linewidths=0, rasterized=True)
        ax.set_xticks(list(range(-4, 5, 2)))
        ax.set_yticks(list(range(-4, 5, 2)))
        ax.tick_params(axis='both', labelsize=6, pad=0.1)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(linestyle=':')
for j, mode in enumerate(modes):
    axes[0][j].set_title(labels[j], fontsize=10, fontfamily='serif')
for i, dataset in enumerate(datasets_labels):
    axes[i][0].set_ylabel(dataset, rotation=90, fontsize=10, fontfamily='serif')
plt.savefig('simulation1-revise.png', dpi=300, pad_inches=0., bbox_inches='tight')
