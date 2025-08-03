import argparse
import os
import pickle

import flexcode
import nnkcde
import numpy as np
from cdetools.cde_loss import cde_loss
from flexcode.regression_models import NN
from scipy.integrate import simps

from misc import add_dict_to_argparser, default_logger


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(seed=42, mode='nnkcde', output='logs/', desc="", # meta
                    data='dataset/m1.pkl', n_train=10000, # data
                    ngrid=1000, # for density estimation
                    )
    add_dict_to_argparser(parser, defaults)
    return parser

def calc_mean_and_std(cde, grid):
    ntest = cde.shape[0]
    mean = np.empty(ntest)
    std = np.empty(ntest)
    for i in range(ntest):
        mean[i] = simps(cde[i, :]*grid, x=grid)
        std[i] = simps(cde[i, :]*(grid-mean[i])**2, x=grid)
    return mean, np.sqrt(std)

def run_nnkcde(args, data):
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)

    # normalized
    x_train, y_train = data['x_train'], data['y_train']
    x_valid, y_valid = data['x_valid'], data['y_valid']
    x_test, y_test = data['x_test'], data['y_test']
    # un-normalized
    mean_test, std_test = data['mean_test'], data['std_test'] 

    model = nnkcde.NNKCDE()
    model.fit(y_train, x_train)
    x_grid = np.linspace(x_train.min(), x_train.max(), args.ngrid)

    # Grid search best parameters.
    bw_search_vec = np.linspace(0.001, 0.01, 10)
    k_search_vec = [3, 6, 9]
    results_search = {}
    for bw in bw_search_vec:
        for k in k_search_vec:
            cde_val_temp = model.predict(y_valid, x_grid, k=k, bandwidth=bw)
            cde_loss_temp, std_loss_temp = cde_loss(cde_val_temp, x_grid, x_valid)
            results_search[(bw, k)] = (cde_loss_temp, std_loss_temp)
    best_combination = sorted(results_search.items(), key=lambda x: x[1][0])[0]
    logger.info(f'Best CDE loss ({best_combination[1][0]:4.3f}) with {best_combination[0][1]:d} Neighbors and KDE bandwidth={best_combination[0][0]:.3f}')
    best_k, best_bw = best_combination[0][1], best_combination[0][0]

    # Prediction
    cde = model.predict(y_test, x_grid, k=best_k, bandwidth=best_bw)
    mean_pred, std_pred = calc_mean_and_std(cde, x_grid)
    logger.info(f"MSE(mean) distance: {np.mean(np.square(mean_pred[:, None] - mean_test)):.3f}")
    logger.info(f"MSE(std) distance: {np.mean(np.square(std_pred[:, None] - std_test)):.3f}")


def run_flexcode(args, data):
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)

    x_train, y_train = data['x_train'], data['y_train']
    x_valid, y_valid = data['x_valid'], data['y_valid']
    x_test, y_test = data['x_test'], data['y_test']
    mean_test, std_test = data['mean_test'], data['std_test'] # un-normalized

    model = nnkcde.NNKCDE()
    model.fit(y_train, x_train)

    basis_system = "cosine"
    max_basis = 31
    params = {"k": [3, 6, 9]} 
    model = flexcode.FlexCodeModel(NN, max_basis, basis_system, regression_params=params)
    model.fit(y_train, x_train)
    model.tune(y_valid, x_valid, n_grid=args.ngrid)

    cde, x_grid = model.predict(y_test, n_grid=args.ngrid)
    mean_pred, std_pred = calc_mean_and_std(cde, x_grid.squeeze())
    logger.info(f"MSE(mean) distance: {np.mean(np.square(mean_pred[:, None] - mean_test)):.3f}")
    logger.info(f"MSE(std) distance: {np.mean(np.square(std_pred[:, None] - std_test)):.3f}")


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
    x_train, y_train = data['x_train'].cpu().numpy(), data['y_train'].cpu().numpy()
    x_test, y_test = data['x_test'].cpu().numpy(), data['y_test'].cpu().numpy()
    x_valid, y_valid = data['x_valid'].cpu().numpy(), data['y_valid'].cpu().numpy()
    mean_test, std_test = data['mean_test'].cpu().numpy(), data['std_test'].cpu().numpy()
    x_train, y_train = x_train[:args.n_train], y_train[:args.n_train]
    data = dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_valid=x_valid, y_valid=y_valid, mean_test=mean_test, std_test=std_test)

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
