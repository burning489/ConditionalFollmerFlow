import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


datasets = ['4squares', 'checkerboard', 'pinwheel', 'swissroll']
dataset_labels = ['4 squares', 'checkerboard', 'pinwheel', 'Swiss roll']
T_list = [0.999, 0.9995, 0.9999]

table = {}
for dataset in datasets:
    table[dataset] = {}
    for T in T_list:
        data = np.load(f'logs/{dataset}-{T}.npz')
        table[dataset][T] = data

fig, axes = plt.subplots(len(datasets), len(T_list), figsize=(5, 5), gridspec_kw={'wspace': 0.1, 'hspace': 0.2}, layout=None)
for i, dataset in enumerate(datasets):
    for j, T in enumerate(T_list):
        ax = axes[i][j]
        data = table[dataset][T]
        x, y = data['x'], data['y']
        ax.scatter(x, y, s=0.5, marker='o', facecolors='C0', edgecolors='none', linewidths=0, rasterized=True)
        ax.set_xticks(list(range(-4, 5, 2)))
        ax.set_yticks(list(range(-4, 5, 2)))
        ax.tick_params(axis='both', labelsize=6, pad=0.1)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(linestyle=':')
for i, dataset in enumerate(dataset_labels):
    axes[i][0].set_ylabel(dataset, rotation=90, fontsize=10, fontfamily='serif')
for j, T in enumerate(T_list):
    axes[0][j].set_title(f'T={T}', fontsize=10, fontfamily='serif')
plt.savefig('simulation1-T.png', dpi=300, pad_inches=0., bbox_inches='tight')
