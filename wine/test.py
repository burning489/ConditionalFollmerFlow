import argparse
import os
import pickle

import numpy as np
import torch
from scipy.stats import t

from diffusion import Follmer, Linear, VESDE, Trigonometric
from misc import add_dict_to_argparser, default_logger


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(device=0, seed=7, mode='follmer', ckpt='', # meta
                    data='dataset/wine.pkl', # data
                    latent_dim=50, # VAE/GAN
                    eps=0, T=0.999, N=64, # diffusion
                    ode_solver='euler', # flow-based specific
                    sigma_min=0.1, sigma_max=100, sde_solver='euler-maruyama', # VE SDE specific
                    bsz=1000, # number of generated samples for each condition
                    alpha=0.1 # coverage
                    )
    add_dict_to_argparser(parser, defaults)
    return parser


def test_diffusion(args):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    logger = default_logger(os.path.join(rundir, f'test-alpha{args.alpha}.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'ckpt: {args.ckpt}')
    logger.info(f'data_name: {data_name}')
    logger.info(f'ode solver: {args.ode_solver}')
    logger.info(f'sde solver: {args.sde_solver}')
    logger.info(f'eps: {args.eps}')
    logger.info(f'T: {args.T}')
    logger.info(f'N: {args.N}')

    # Load data.
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x_test'].to(device), data['y_test'].to(device)
    logger.info(f'n_test: {x.shape[0]}')
    args.data_dim = x.shape[-1]
    args.cond_dim = y.shape[-1]
    n_test = x.shape[0]

    # Network and diffusion.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()
    if args.mode == 'follmer':
        sde = Follmer(args)
    elif args.mode == 'linear':
        sde = Linear(args)
    elif args.mode == 'trig':
        sde = Trigonometric(args)
    elif args.mode == 'vesde':
        sde = VESDE(args)
    else:
        raise ValueError(f'unrecognized mode {args.mode}')

    # Precision Benchmark.
    y_tgt = y.repeat(args.bsz, 1) # (n_test * bsz, cond_dim)
    noise = sde.prior_sampling(shape=[y_tgt.shape[0], args.data_dim], device=device)
    x_pred = sde.solve(model, noise, y_tgt)
    x_pred = x_pred.cpu().numpy().flatten() # (n_test * bsz, )
    x_test = x.cpu().numpy().flatten()

    lower_bound = np.empty(n_test)
    upper_bound = np.empty(n_test)
    t_value = t.ppf(1 - args.alpha/ 2, df=args.bsz - 1)
    for i in range(n_test):
        pred_mean = np.mean(x_pred[i::n_test])
        residual_std = np.std(x_pred[i::n_test] - x_test[i])
        se = residual_std * np.sqrt(1 + 1/args.bsz)
        lower_bound[i] = pred_mean - t_value * se
        upper_bound[i] = pred_mean + t_value * se
    accuracy = np.sum((lower_bound <= x_test) & (x_test <= upper_bound)) / n_test
    logger.info(f'Coverage distance: {accuracy:.4f}')


def test_vae(args):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    logger = default_logger(os.path.join(rundir, f'test-alpha{args.alpha}.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'ckpt: {args.ckpt}')
    logger.info(f'data_name: {data_name}')
    logger.info(f'latent_dim: {args.latent_dim}')

    # Load data.
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x_test'].to(device), data['y_test'].to(device)
    logger.info(f'n_test: {x.shape[0]}')
    args.data_dim = x.shape[-1]
    args.cond_dim = y.shape[-1]
    n_test = x.shape[0]

    # Model.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    # Precision Benchmark.
    y_tgt = y.repeat(args.bsz, 1) # (n_test * bsz, cond_dim)
    noise = torch.randn((y_tgt.shape[0], args.latent_dim), device=device)
    with torch.no_grad():
        x_pred = model.decoder(noise, y_tgt)
    x_pred = x_pred.cpu().numpy().flatten() # (n_test * bsz, )
    x_test = x.cpu().numpy().flatten()

    lower_bound = np.empty(n_test)
    upper_bound = np.empty(n_test)
    t_value = t.ppf(1 - args.alpha/ 2, df=args.bsz - 1)
    for i in range(n_test):
        pred_mean = np.mean(x_pred[i::n_test])
        residuals = x_pred[i::n_test] - x_test[i]
        residual_std = np.std(residuals)
        se = residual_std * np.sqrt(1 + 1/args.bsz)
        lower_bound[i] = pred_mean - t_value * se
        upper_bound[i] = pred_mean + t_value * se
    accuracy = np.sum((lower_bound <= x_test) & (x_test <= upper_bound)) / n_test
    logger.info(f'Coverage distance: {accuracy:.4f}')


def test_gan(args):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    logger = default_logger(os.path.join(rundir, f'test-alpha{args.alpha}.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'ckpt: {args.ckpt}')
    logger.info(f'data_name: {data_name}')
    logger.info(f'latent_dim: {args.latent_dim}')

    # Load data.
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x_test'].to(device), data['y_test'].to(device)
    logger.info(f'n_test: {x.shape[0]}')
    args.data_dim = x.shape[-1]
    args.cond_dim = y.shape[-1]
    n_test = x.shape[0]

    # Model.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    # Precision Benchmark.
    y_tgt = y.repeat(args.bsz, 1) # (n_test * bsz, cond_dim)
    noise = torch.randn((y_tgt.shape[0], args.latent_dim), device=device)
    with torch.no_grad():
        x_pred = model.generator(noise, y_tgt)
    x_pred = x_pred.cpu().numpy().flatten() # (n_test * bsz, )
    x_test = x.cpu().numpy().flatten()
    
    lower_bound = np.empty(n_test)
    upper_bound = np.empty(n_test)
    t_value = t.ppf(1 - args.alpha/ 2, df=args.bsz - 1)
    for i in range(n_test):
        pred_mean = np.mean(x_pred[i::n_test])
        residuals = x_pred[i::n_test] - x_test[i]
        residual_std = np.std(residuals)
        se = residual_std * np.sqrt(1 + 1/args.bsz)
        lower_bound[i] = pred_mean - t_value * se
        upper_bound[i] = pred_mean + t_value * se
    accuracy = np.sum((lower_bound <= x_test) & (x_test <= upper_bound)) / n_test
    logger.info(f'Coverage distance: {accuracy:.4f}')

if __name__ == '__main__':
    args = create_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode in ['follmer', 'linear', 'trig', 'vesde']:
        test_diffusion(args)
    elif args.mode == 'vae':
        test_vae(args)
    elif args.mode == 'gan':
        test_gan(args)
    else:
        raise ValueError(f"unsupported eval mode {args.mode}")
