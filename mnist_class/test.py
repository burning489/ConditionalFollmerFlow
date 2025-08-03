import argparse
import os

import numpy as np
import torch

from diffusion import Follmer, Linear, VESDE, Trigonometric
from misc import add_dict_to_argparser, default_logger


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(device=0, seed=42, mode='follmer', ckpt='', # meta
                    img_res=28, in_ch=1, out_ch=1, num_classes=10,
                    latent_dim=200, # VAE/GAN
                    eps=1e-3, T=0.999, N=64, # diffusion
                    ode_solver='euler', # flow-based specific
                    sigma_min=0.1, sigma_max=100, sde_solver='euler-maruyama', # VE SDE specific
                    n_per_class=5000, # number of samples for each class
                    bsz=500
                    )
    add_dict_to_argparser(parser, defaults)
    return parser


def test_diffusion(args):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    logger = default_logger(os.path.join(rundir, f'test.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'ckpt: {args.ckpt}')
    logger.info(f'ode solver: {args.ode_solver}')
    logger.info(f'sde solver: {args.sde_solver}')
    logger.info(f'eps: {args.eps}')
    logger.info(f'T: {args.T}')
    logger.info(f'N: {args.N}')
    logger.info(f'samples per class: {args.n_per_class}')
    logger.info(f'batch size: {args.bsz}')

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

    n_tot = args.n_per_class * args.num_classes
    x = np.empty((n_tot, args.out_ch, args.img_res, args.img_res))
    for class_idx in range(args.num_classes):
        y = torch.eye(args.num_classes, device=device)[[class_idx, ]*args.bsz]
        for i in range(0, args.n_per_class, args.bsz):
            noise = sde.prior_sampling(shape=[args.bsz, args.in_ch, args.img_res, args.img_res], device=device)
            xi = sde.solve(model, noise, y)
            start = class_idx * args.n_per_class + i
            stop = start + args.bsz
            x[start:stop] = xi.cpu().numpy()
            logger.info(f'class_idx [{class_idx+1:2d}/{args.num_classes:2d}], sample [{i+args.bsz:2d}/{args.n_per_class:2d}]')
    x = (x * 127.5 + 128).clip(0, 255).astype(np.uint8) # [N, C, H, W]
    np.save(os.path.join(rundir, f'samples.npy'), x)


def test_vae(args):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    logger = default_logger(os.path.join(rundir, f'test.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'ckpt: {args.ckpt}')
    logger.info(f'latent_dim: {args.latent_dim}')
    logger.info(f'samples per class: {args.n_per_class}')
    logger.info(f'batch size: {args.bsz}')

    # Model.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    n_tot = args.n_per_class * args.num_classes
    x = np.empty((n_tot, args.out_ch, args.img_res, args.img_res))
    with torch.no_grad():
        for class_idx in range(args.num_classes):
            y = torch.eye(args.num_classes, device=device)[[class_idx, ]*args.bsz]
            for i in range(0, args.n_per_class, args.bsz):
                noise = torch.randn((args.bsz, args.latent_dim), device=device)
                xi = model.decoder(noise, y)
                start = class_idx * args.n_per_class + i
                stop = start + args.bsz
                x[start:stop] = xi.cpu().numpy()
                logger.info(f'class_idx [{class_idx+1:2d}/{args.num_classes:2d}], sample [{i+args.bsz:2d}/{args.n_per_class:2d}]')
    x = (x * 127.5 + 128).clip(0, 255).astype(np.uint8) # [N, C, H, W]
    np.save(os.path.join(rundir, 'samples.npy'), x)


def test_gan(args):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    logger = default_logger(os.path.join(rundir, f'test.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'ckpt: {args.ckpt}')
    logger.info(f'latent_dim: {args.latent_dim}')

    # Model.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    n_tot = args.n_per_class * args.num_classes
    x = np.empty((n_tot, args.out_ch, args.img_res, args.img_res))
    with torch.no_grad():
        for class_idx in range(args.num_classes):
            y = torch.eye(args.num_classes, device=device)[[class_idx, ]*args.bsz]
            for i in range(0, args.n_per_class, args.bsz):
                noise = torch.randn((args.bsz, args.latent_dim), device=device)
                xi = model.generator(noise, y)
                start = class_idx * args.n_per_class + i
                stop = start + args.bsz
                x[start:stop] = xi.cpu().numpy()
                logger.info(f'class_idx [{class_idx+1:2d}/{args.num_classes:2d}], sample [{i+args.bsz:2d}/{args.n_per_class:2d}]')
    x = (x * 127.5 + 128).clip(0, 255).astype(np.uint8) # [N, C, H, W]
    np.save(os.path.join(rundir, 'samples.npy'), x)


if __name__ == '__main__':
    args = create_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode in ['follmer', 'linear', 'vesde', 'trig']:
        test_diffusion(args)
    elif args.mode == 'vae':
        test_vae(args)
    elif args.mode == 'gan':
        test_gan(args)
    else:
        raise ValueError(f"unsupported eval mode {args.mode}")
