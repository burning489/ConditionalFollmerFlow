import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import simps
from sklearn.neighbors import KernelDensity

# from data import generate_toy_data
from diffusion import Follmer, Trigonometric, VESDE, Linear
from misc import add_dict_to_argparser, default_logger


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(device=0, seed=7, mode='follmer', ckpt='', # meta
                    n_rpt=100, # benchmark repeats
                    data='dataset/moons.pkl', data_dim=1, cond_dim=1, # data
                    latent_dim=32, # VAE/GAN
                    eps=1e-3, T=0.999, N=100, # diffusion
                    ode_solver='euler', # flow-based specific
                    sigma_min=0.25, sigma_max=100, sde_solver='euler-maruyama', # VE SDE specific
                    bsz=500, n_grid=200, bandwidth='scott' # conditional density estimation
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
    x_test, y_test = data['x_test'], data['y_test']
    x_test, y_test = torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device)
    logger.info(f'n_test: {x_test.shape[0]}')

    # Network and diffusion.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()
    if args.mode == 'follmer':
        sde = Follmer(args)
    elif args.mode == 'trig':
        sde = Trigonometric(args)
    elif args.mode == 'linear':
        sde = Linear(args)
    elif args.mode == 'vesde':
        sde = VESDE(args)
    else:
        raise ValueError(f'unrecognized mode {args.mode}')

    # Speed Benchmark.
    noise = sde.prior_sampling(shape=[y_test.shape[0], args.data_dim], device=device)
    tic = time.time()
    # for _ in range(args.n_rpt):
    for _ in range(0):
        _ = sde.solve(model, noise, y_test)
    toc = time.time()
    lapsed = (toc - tic) / args.n_rpt
    logger.info(f"avg. duration (repeated {args.n_rpt} times): {lapsed:.3f} sec.")

    # Precision Benchmark.
    ## Reference density estimation (on normalized data).
    xy = torch.hstack([x_test, y_test]).cpu().numpy()
    ref_kde = KernelDensity(bandwidth=args.bandwidth)
    ref_kde.fit(xy)
    grid = np.empty((args.n_grid, 2))
    x_grid = np.linspace(-4, 4, args.n_grid)
    grid[:, 0] = x_grid

    ## Predict (normalized)
    y_tgt = y_test[:args.n_rpt].repeat(args.bsz, 1)
    noise = sde.prior_sampling(shape=[args.n_rpt * args.bsz, args.data_dim], device=device)
    x_pred = sde.solve(model, noise, y_tgt).cpu().numpy()

    ## Loop over n_rpt conditions.
    metrics = np.empty(args.n_rpt)
    for i in range(args.n_rpt):
        # reference conditional density given i-th condition
        grid[:, 1] = y_tgt.cpu().numpy()[i]
        qk = np.exp(ref_kde.score_samples(grid))
        qk = qk / simps(qk, x_grid)
        # collect generated samples under the same condition
        pred_i = x_pred[i::args.n_rpt]
        # marginal density estimation on generated samples
        kde = KernelDensity(bandwidth=args.bandwidth)
        kde.fit(pred_i)
        pk = np.exp(kde.score_samples(x_grid[:, None]))
        metrics[i] = simps(np.abs(qk - pk), x_grid) / 2
    logger.info(f"TV distance: {np.mean(metrics):.3f}(pm {np.std(metrics):.3f})")

    # Visualization.
    noise = sde.prior_sampling(shape=[y_test.shape[0], args.data_dim], device=device)
    x_pred = sde.solve(model, noise, y_test)
    x_pred = x_pred.cpu().numpy()
    y_pred = y_test.cpu().numpy()
    plt.figure(figsize=(2, 2))
    plt.scatter(x_pred, y_pred, s=0.1)
    plt.xticks(list(range(-4, 5, 2)))
    plt.yticks(list(range(-4, 5, 2)))
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(rundir, f'{args.mode}.png'), dpi=300)
    plt.close()

    np.savez(os.path.join(rundir, "samples.npz"), x=x_pred, y=y_pred)

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
    x_test, y_test = data['x_test'], data['y_test']
    x_test, y_test = torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device)
    logger.info(f'n_test: {x_test.shape[0]}')

    # Model.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    # Speed Benchmark.
    z = torch.randn((y_test.shape[0], args.latent_dim), device=device)
    tic = time.time()
    with torch.no_grad():
        # for _ in range(args.n_rpt):
        for _ in range(0):
            _ = model.decoder(z, y_test)
    toc = time.time()
    lapsed = (toc - tic) / args.n_rpt
    logger.info(f"avg. duration (repeated {args.n_rpt} times): {lapsed:.3f} sec.")

    # Precision Benchmark.
    ## Reference density estimation (on normalized data).
    xy = torch.hstack([x_test, y_test]).cpu().numpy()
    ref_kde = KernelDensity(bandwidth=args.bandwidth)
    ref_kde.fit(xy)
    grid = np.empty((args.n_grid, 2))
    x_grid = np.linspace(-4, 4, args.n_grid)
    grid[:, 0] = x_grid

    ## Predict (normalized)
    y_tgt = y_test[:args.n_rpt].repeat(args.bsz, 1)
    z = torch.randn((args.n_rpt * args.bsz, args.latent_dim), device=device)
    with torch.no_grad():
        x_pred = model.decoder(z, y_tgt).cpu().numpy()

    ## Loop over n_rpt conditions.
    metrics = np.empty(args.n_rpt)
    for i in range(args.n_rpt):
        # reference conditional density given i-th condition
        grid[:, 1] = y_tgt.cpu().numpy()[i]
        qk = np.exp(ref_kde.score_samples(grid))
        qk = qk / simps(qk, x_grid)
        # collect generated samples under the same condition
        pred_i = x_pred[i::args.n_rpt]
        # marginal density estimation on generated samples
        kde = KernelDensity(bandwidth=args.bandwidth)
        kde.fit(pred_i)
        pk = np.exp(kde.score_samples(x_grid[:, None]))
        metrics[i] = simps(np.abs(qk - pk), x_grid) / 2
    logger.info(f"TV distance: {np.mean(metrics):.3f}(pm {np.std(metrics):.3f})")

    # Visualization.
    z = torch.randn((y_test.shape[0], args.latent_dim), device=device)
    with torch.no_grad():
        x_pred = model.decoder(z, y_test)
    x_pred = x_pred.cpu().numpy()
    y_pred = y_test.cpu().numpy()
    plt.figure(figsize=(2, 2))
    plt.scatter(x_pred, y_pred, s=0.1)
    plt.xticks(list(range(-4, 5, 2)))
    plt.yticks(list(range(-4, 5, 2)))
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(rundir, 'vae.png'), dpi=300)
    plt.close()

    np.savez(os.path.join(rundir, "samples.npz"), x=x_pred, y=y_pred)


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
    x_test, y_test = data['x_test'], data['y_test']
    x_test, y_test = torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device)
    logger.info(f'n_test: {x_test.shape[0]}')

    # Model.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    # Speed Benchmark.
    z = torch.randn((y_test.shape[0], args.latent_dim), device=device)
    tic = time.time()
    with torch.no_grad():
        # for _ in range(args.n_rpt):
        for _ in range(0):
            _ = model.generator(z, y_test)
    toc = time.time()
    lapsed = (toc - tic) / args.n_rpt
    logger.info(f"avg. duration (repeated {args.n_rpt} times): {lapsed:.3f} sec.")

    # Precision Benchmark.
    ## Reference density estimation (on normalized data).
    xy = torch.hstack([x_test, y_test]).cpu().numpy()
    ref_kde = KernelDensity(bandwidth=args.bandwidth)
    ref_kde.fit(xy)
    grid = np.empty((args.n_grid, 2))
    x_grid = np.linspace(-4, 4, args.n_grid)
    grid[:, 0] = x_grid

    ## Predict (normalized)
    y_tgt = y_test[:args.n_rpt].repeat(args.bsz, 1)
    z = torch.randn((args.n_rpt * args.bsz, args.latent_dim), device=device)
    with torch.no_grad():
        x_pred = model.generator(z, y_tgt).cpu().numpy()

    ## Loop over n_rpt conditions.
    metrics = np.empty(args.n_rpt)
    for i in range(args.n_rpt):
        # reference conditional density given i-th condition
        grid[:, 1] = y_tgt.cpu().numpy()[i]
        qk = np.exp(ref_kde.score_samples(grid))
        qk = qk / simps(qk, x_grid)
        # collect generated samples under the same condition
        pred_i = x_pred[i::args.n_rpt]
        # marginal density estimation on generated samples
        kde = KernelDensity(bandwidth=args.bandwidth)
        kde.fit(pred_i)
        pk = np.exp(kde.score_samples(x_grid[:, None]))
        metrics[i] = simps(np.abs(qk - pk), x_grid) / 2
    logger.info(f"TV distance: {np.mean(metrics):.3f}(pm {np.std(metrics):.3f})")

    # Visualization.
    z = torch.randn((y_test.shape[0], args.latent_dim), device=device)
    with torch.no_grad():
        x_pred = model.generator(z, y_test)
    x_pred = x_pred.cpu().numpy()
    y_pred = y_test.cpu().numpy()
    plt.figure(figsize=(2, 2))
    plt.scatter(x_pred, y_pred, s=0.1)
    plt.xticks(list(range(-4, 5, 2)))
    plt.yticks(list(range(-4, 5, 2)))
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(linestyle=':')
    plt.savefig(os.path.join(rundir, 'gan.png'), dpi=300)
    plt.close()

    np.savez(os.path.join(rundir, "samples.npz"), x=x_pred, y=y_pred)


if __name__ == '__main__':
    args = create_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode in ['follmer', 'trig', 'linear', 'vesde']:
        test_diffusion(args)
    elif args.mode == 'vae':
        test_vae(args)
    elif args.mode == 'gan':
        test_gan(args)
    else:
        raise ValueError(f"unsupported eval mode {args.mode}")
