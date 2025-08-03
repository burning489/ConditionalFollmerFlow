import argparse
import os
import pickle

import flexcode
import matplotlib.pyplot as plt
import numpy as np
from cdetools.cde_loss import cde_loss
from flexcode.regression_models import NN
from scipy.integrate import simps
from sklearn.neighbors import KernelDensity

import nnkcde
from misc import add_dict_to_argparser, default_logger


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(seed=42, mode='nnkcde', output='logs/', desc="",  # meta
                    data='dataset/moons.pkl', n_train=50000,  # data
                    ngrid=200,  # conditional density estimation
                    n_rpt=100  # benchmark repeats
                    )
    add_dict_to_argparser(parser, defaults)
    return parser


def run_nnkcde(args, data):
    model = nnkcde.NNKCDE()
    model.fit(data['y_train'], data['x_train'])
    x_grid = np.linspace(-4, 4, args.ngrid)

    # grid search best parameters
    bw_search_vec = np.linspace(0.01, 0.1, 10)
    k_search_vec = [3, 6, 9]
    results_search = {}
    for bw in bw_search_vec:
        for k in k_search_vec:
            cde_test_temp = model.predict(data['y_valid'], x_grid, k=k, bandwidth=bw)
            cde_loss_temp, std_loss_temp = cde_loss(cde_test_temp, x_grid, data['x_valid'])
            results_search[(bw, k)] = (cde_loss_temp, std_loss_temp)
    best_combination = sorted(results_search.items(), key=lambda x: x[1][0])[0]
    logger.info(
        f'Best CDE loss ({best_combination[1][0]:4.3f})'
        f'with {best_combination[0][1]:d} Neighbors, KDE bandwidth={best_combination[0][0]:.3f}')
    best_k, best_bw = best_combination[0][1], best_combination[0][0]

    y_test = data['y_test'][:args.n_rpt]
    cde_test = model.predict(y_test, x_grid, k=best_k, bandwidth=best_bw)
    kde = KernelDensity(kernel="gaussian", bandwidth='scott')
    kde.fit(np.hstack((data['x_train'], data['y_train'])))
    grid = np.empty((args.ngrid, 2))
    grid[:, 0] = x_grid
    metrics = np.empty(args.n_rpt)
    for i in range(args.n_rpt):
        pk = cde_test[i, :]
        grid[:, 1] = y_test[i]
        qk = np.exp(kde.score_samples(grid))
        qk = qk / simps(qk, x_grid)
        metrics[i] = simps(np.abs(qk - pk), x_grid) / 2
    logger.info(f"TV distance: {np.mean(metrics):.3f}(pm {np.std(metrics):.3f})")

    y_plot = data['y_test']
    x_plot = np.empty_like(y_plot)
    cde_plot = model.predict(data['y_test'], x_grid, k=best_k, bandwidth=best_bw)
    for i in range(x_plot.shape[0]):
        x_plot[i] = np.random.choice(x_grid, p=cde_plot[i, :] / cde_plot[i, :].sum())
    plt.figure(figsize=(2, 2))
    plt.scatter(x_plot, y_plot, s=0.1)
    plt.xticks(list(range(-4, 5, 2)))
    plt.yticks(list(range(-4, 5, 2)))
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(args.rundir, f"nnkcde.png"), dpi=300)
    plt.close()

    np.savez(os.path.join(args.rundir, "samples.npz"), x=x_plot, y=y_plot)


def run_flexcode(args, data):
    model = nnkcde.NNKCDE()
    model.fit(data['y_train'], data['x_train'])

    basis_system = "cosine"
    max_basis = 31
    params = {"k": [3, 6, 9]}
    model = flexcode.FlexCodeModel(NN, max_basis, basis_system, regression_params=params)
    model.fit(data['y_train'], data['x_train'])
    model.tune(data['y_valid'], data['x_valid'], n_grid=args.ngrid)

    y_test = data['y_test'][:args.n_rpt]
    cde_test, x_grid = model.predict(y_test, n_grid=args.ngrid)
    kde = KernelDensity(kernel="gaussian", bandwidth='scott')
    kde.fit(np.hstack((data['x_train'], data['y_train'])))
    grid = np.empty((args.ngrid, 2))
    grid[:, 0] = x_grid.squeeze()
    metrics = np.empty(args.n_rpt)
    for i in range(args.n_rpt):
        pk = cde_test[i, :]
        grid[:, 1] = y_test[i]
        qk = np.exp(kde.score_samples(grid))
        qk = qk / simps(qk, x_grid.squeeze())
        metrics[i] = simps(np.abs(qk - pk), x_grid.squeeze()) / 2
    logger.info(f"TV distance: {np.mean(metrics):.3f}(pm {np.std(metrics):.3f})")

    y_plot = data['y_test']
    x_plot = np.empty_like(y_plot)
    cde_plot, x_grid = model.predict(y_plot, n_grid=args.ngrid)
    for i in range(x_plot.shape[0]):
        x_plot[i] = np.random.choice(x_grid.squeeze(), p=cde_plot[i, :] / cde_plot[i, :].sum())
    plt.figure(figsize=(2, 2))
    plt.scatter(x_plot, y_plot, s=0.1)
    plt.xticks(list(range(-4, 5, 2)))
    plt.yticks(list(range(-4, 5, 2)))
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(args.rundir, f"flexcode.png"), dpi=300)
    plt.close()

    np.savez(os.path.join(args.rundir, "samples.npz"), x=x_plot, y=y_plot)


if __name__ == '__main__':
    # Meta.
    args = create_parser().parse_args()
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    desc = f'{data_name:s}-n{args.n_train}-{args.mode}'
    if args.desc:
        desc += f'-{args.desc}'
    rundir = os.path.join(args.output, desc)
    os.makedirs(rundir, exist_ok=True)
    args.rundir = rundir
    np.random.seed(args.seed)
    print(args.rundir)

    # Load data.
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    x_valid, y_valid = data['x_valid'], data['y_valid']
    x_train, y_train = x_train[:args.n_train], y_train[:args.n_train]
    data = dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_valid=x_valid, y_valid=y_valid)

    # Logging
    logger = default_logger(os.path.join(args.rundir, 'test.log'))
    logger.info(f'dataset: {data_name}')
    logger.info(f'n_train: {args.n_train}')

    if args.mode == 'nnkcde':
        run_nnkcde(args, data)
    elif args.mode == 'flexcode':
        run_flexcode(args, data)
    else:
        raise ValueError(f'unrecognized mode {args.mode}')
