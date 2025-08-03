import argparse
import os
import pickle

from ucimlrepo import fetch_ucirepo
import numpy as np
import torch
from misc import add_dict_to_argparser

def load_wine_dataset():
    wine_quality_dataset = fetch_ucirepo(id=186)
    x = wine_quality_dataset.data.targets.to_numpy()
    y = wine_quality_dataset.data.features.to_numpy()
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    defaults = dict(data='m1', n_train=5847, n_test=650, seed=42, output='dataset/wine.pkl')
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()
    np.random.seed(args.seed)
    x, y = load_wine_dataset()
    indices = np.random.permutation(x.shape[0])
    x, y = x[indices], y[indices]
    x_min, x_max, y_min, y_max = x.min(axis=0), x.max(axis=0), y.min(axis=0), y.max(axis=0)
    x_mean, x_std, y_mean, y_std = x.mean(axis=0), x.std(axis=0), y.mean(axis=0), y.std(axis=0)
    x = (x - x_min) / (x_max - x_min)
    x = 2*x-1
    y = (y - y_min) / (y_max - y_min)
    y = 2*y-1
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    x_train, y_train = x[:args.n_train], y[:args.n_train]
    x_test, y_test = x[-args.n_test:], y[-args.n_test:]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(dict(
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
        ), f)