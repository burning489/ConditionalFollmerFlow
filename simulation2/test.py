import argparse
import os
import pickle
import time

import numpy as np
import torch

from diffusion import Follmer, Linear, VESDE, Trigonometric
from misc import add_dict_to_argparser, default_logger


EVAL_BSZ = 200000

def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(device=0, seed=39, mode='follmer', ckpt='', # meta
                    n_rpt=500, # benchmark repeats
                    data='dataset/m1.pkl', # data
                    latent_dim=100, # VAE/GAN
                    eps=1e-3, T=0.999, N=100, # diffusion
                    ode_solver='euler', # flow-based specific
                    sigma_min=0.25, sigma_max=100, sde_solver='euler-maruyama', # VE SDE specific
                    bsz=500, # number of generated samples for each condition
                    )
    add_dict_to_argparser(parser, defaults)
    return parser


def test_diffusion(args):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    logger = default_logger(os.path.join(rundir, 'test.log'))
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
    mean_test, std_test = data['mean_test'].cpu().numpy(), data['std_test'].cpu().numpy()
    logger.info(f'n_test: {y.shape[0]}')
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

    # Speed Benchmark.
    noise = sde.prior_sampling(shape=[n_test, args.data_dim], device=device)
    tic = time.time()
    for _ in range(0):
        _ = sde.solve(model, noise, y)
    toc = time.time()
    lapsed = (toc - tic) / args.n_rpt
    logger.info(f"avg. duration (repeated {args.n_rpt} times): {lapsed:.3f} sec.")

    # Precision Benchmark.
    y_tgt = y.repeat(args.bsz, 1) # (n_test * bsz, cond_dim)
    noise = sde.prior_sampling(shape=[y_tgt.shape[0], args.data_dim], device=device)
    x_pred = np.empty((y_tgt.shape[0], args.data_dim))
    for start in range(0, y_tgt.shape[0], EVAL_BSZ):
        end = min(start + EVAL_BSZ, y_tgt.shape[0])
        x = sde.solve(model, noise[start:end, :], y_tgt[start:end, :])
        x_pred[start:end, :] = x.cpu().numpy()
    x_pred = x_pred.reshape(args.bsz, n_test, 1)
    mean_pred = np.mean(x_pred, axis=0)
    std_pred = np.std(x_pred, axis=0)
    logger.info(f"MSE(mean) distance: {np.mean(np.square(mean_pred - mean_test)):.3f}")
    logger.info(f"MSE(std) distance: {np.mean(np.square(std_pred - std_test)):.3f}")


def test_vae(args):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    logger = default_logger(os.path.join(rundir, 'test.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'ckpt: {args.ckpt}')
    logger.info(f'data_name: {data_name}')
    logger.info(f'latent_dim: {args.latent_dim}')

    # Load data.
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x_test'].to(device), data['y_test'].to(device)
    mean_test, std_test = data['mean_test'].cpu().numpy(), data['std_test'].cpu().numpy()
    logger.info(f'n_test: {x.shape[0]}')
    args.data_dim = x.shape[-1]
    args.cond_dim = y.shape[-1]
    n_test = x.shape[0]

    # Model
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    # Speed Benchmark.
    z = torch.randn((n_test, args.latent_dim), device=device)
    tic = time.time()
    with torch.no_grad():
        for _ in range(0):
            _ = model.decoder(z, y)
    toc = time.time()
    lapsed = (toc - tic) / args.n_rpt
    logger.info(f"avg. duration (repeated {args.n_rpt} times): {lapsed:.3f} sec.")

    # Precision Benchmark.
    y_tgt = y.repeat(args.bsz, 1) # (n_test * bsz, cond_dim)
    noise = torch.randn((y_tgt.shape[0], args.latent_dim), device=device)
    x_pred = np.empty((y_tgt.shape[0], args.data_dim))
    with torch.no_grad():
        for start in range(0, y_tgt.shape[0], EVAL_BSZ):
            end = min(start + EVAL_BSZ, y_tgt.shape[0])
            x = model.decoder(noise[start:end, :], y_tgt[start:end, :])
            x_pred[start:end, :] = x.cpu().numpy()
    x_pred = x_pred.reshape(args.bsz, n_test, 1)
    mean_pred = np.mean(x_pred, axis=0)
    std_pred = np.std(x_pred, axis=0)
    logger.info(f"MSE(mean) distance: {np.mean(np.square(mean_pred - mean_test)):.3f}")
    logger.info(f"MSE(std) distance: {np.mean(np.square(std_pred - std_test)):.3f}")


def test_gan(args):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    logger = default_logger(os.path.join(rundir, 'test.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'ckpt: {args.ckpt}')
    logger.info(f'data_name: {data_name}')
    logger.info(f'latent_dim: {args.latent_dim}')

    # Load data.
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x_test'].to(device), data['y_test'].to(device)
    mean_test, std_test = data['mean_test'].cpu().numpy(), data['std_test'].cpu().numpy()
    logger.info(f'n_test: {x.shape[0]}')
    args.data_dim = x.shape[-1]
    args.cond_dim = y.shape[-1]
    n_test = x.shape[0]

    # Model.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    # Speed Benchmark.
    z = torch.randn((n_test, args.latent_dim), device=device)
    tic = time.time()
    with torch.no_grad():
        for _ in range(0):
            _ = model.generator(z, y)
    toc = time.time()
    lapsed = (toc - tic) / args.n_rpt
    logger.info(f"avg. duration (repeated {args.n_rpt} times): {lapsed:.3f} sec.")

    # Precision Benchmark.
    y_tgt = y.repeat(args.bsz, 1) # (n_test * bsz, cond_dim)
    noise = torch.randn((y_tgt.shape[0], args.latent_dim), device=device)
    x_pred = np.empty((y_tgt.shape[0], args.data_dim))
    with torch.no_grad():
        for start in range(0, y_tgt.shape[0], EVAL_BSZ):
            end = min(start + EVAL_BSZ, y_tgt.shape[0])
            x = model.generator(noise[start:end, :], y_tgt[start:end, :])
            x_pred[start:end, :] = x.cpu().numpy()
    x_pred = x_pred.reshape(args.bsz, n_test, 1)
    mean_pred = np.mean(x_pred, axis=0)
    std_pred = np.std(x_pred, axis=0)
    logger.info(f"MSE(mean) distance: {np.mean(np.square(mean_pred - mean_test)):.3f}")
    logger.info(f"MSE(std) distance: {np.mean(np.square(std_pred - std_test)):.3f}")


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
