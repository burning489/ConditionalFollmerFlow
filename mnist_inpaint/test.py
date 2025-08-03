import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import Subset
from torchvision.transforms import ToTensor

from diffusion import Follmer, Linear, VESDE, Trigonometric
from misc import add_dict_to_argparser, default_logger, mask_fn


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(device=0, seed=7, mode='follmer', ckpt='', # meta
                    mask_mode=1, img_res=28, in_ch=1, out_ch=1,
                    latent_dim=50, # VAE/GAN
                    eps=1e-3, T=0.999, N=75, # diffusion
                    ode_solver='euler', # flow-based specific
                    sigma_min=0.1, sigma_max=100, sde_solver='euler-maruyama', # VE SDE specific
                    n_sample=50000, bsz=500
                    )
    add_dict_to_argparser(parser, defaults)
    return parser
    

def test_diffusion(args, loader):
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

    x = np.empty((args.n_sample, args.out_ch, args.img_res, args.img_res))
    cnt = 0
    for _ in range(5):
        for batch in loader:
            img, _ = batch
            y = mask_fn(args.mask_mode)(img)
            img, y = 2*img-1, 2*y-1
            img, y = img.to(device), y.to(device)
            noise = sde.prior_sampling(shape=[img.shape[0], args.in_ch, args.img_res, args.img_res], device=device)
            xi = sde.solve(model, noise, y)
            x[cnt:cnt+img.shape[0]] = xi.cpu().numpy()
            cnt += img.shape[0]
            logger.info(f'image [{cnt:5d}/{args.n_sample:5d}]')
    x = (x * 127.5 + 128).clip(0, 255).astype(np.uint8) # [N, C, H, W]
    np.save(os.path.join(rundir, f'samples.npy'), x)


def test_vae(args, loader):
    # Meta.
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    logger = default_logger(os.path.join(rundir, f'test.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'ckpt: {args.ckpt}')
    logger.info(f'latent_dim: {args.latent_dim}')
    logger.info(f'batch size: {args.bsz}')

    # Model.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    x = np.empty((args.n_sample, args.out_ch, args.img_res, args.img_res))
    cnt = 0
    with torch.no_grad():
        for _ in range(5):
            for batch in loader:
                img, _ = batch
                y = mask_fn(args.mask_mode)(img)
                img, y = 2*img-1, 2*y-1
                img, y = img.to(device), y.to(device)
                noise = torch.randn((img.shape[0], args.latent_dim), device=device)
                xi = model.decoder(noise, y)
                x[cnt:cnt+img.shape[0]] = xi.cpu().numpy()
                cnt += img.shape[0]
                logger.info(f'image [{cnt:5d}/{args.n_sample:5d}]')
    x = (x * 127.5 + 128).clip(0, 255).astype(np.uint8) # [N, C, H, W]
    np.save(os.path.join(rundir, f'samples.npy'), x)


def test_gan(args, loader):
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

    x = np.empty((args.n_sample, args.out_ch, args.img_res, args.img_res))
    cnt = 0
    with torch.no_grad():
        for _ in range(5):
            for batch in loader:
                img, _ = batch
                y = mask_fn(args.mask_mode)(img)
                img, y = 2*img-1, 2*y-1
                img, y = img.to(device), y.to(device)
                noise = torch.randn((img.shape[0], args.latent_dim), device=device)
                xi = model.generator(noise, y)
                x[cnt:cnt+img.shape[0]] = xi.cpu().numpy()
                cnt += img.shape[0]
                logger.info(f'image [{cnt:5d}/{args.n_sample:5d}]')
    x = (x * 127.5 + 128).clip(0, 255).astype(np.uint8) # [N, C, H, W]
    np.save(os.path.join(rundir, f'samples.npy'), x)


if __name__ == '__main__':
    args = create_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = MNIST(
        root="./dataset", 
        transform=ToTensor(),
        train=False)
    # dataset = Subset(dataset, range(args.n_sample))
    loader = DataLoader(dataset, batch_size=args.bsz, shuffle=False, drop_last=False)

    if args.mode in ['follmer', 'linear', 'vesde', 'trig']:
        test_diffusion(args, loader)
    elif args.mode == 'vae':
        test_vae(args, loader)
    elif args.mode == 'gan':
        test_gan(args, loader)
    else:
        raise ValueError(f"unsupported eval mode {args.mode}")
