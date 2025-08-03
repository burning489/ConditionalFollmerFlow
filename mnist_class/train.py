import argparse
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import Subset
import torch.nn.functional as F

from diffusion import Follmer, Linear, VESDE, Trigonometric
from misc import add_dict_to_argparser, default_logger, create_infinite_dataloader
from unet import UNet
from vae import VAE
from gan import GAN


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(device=0, seed=42, output='logs/', desc="", mode="follmer", num_workers=2,  # meta
                    n_train=60000,
                    eps=1e-3, T=0.999, N=100, # diffusion
                    sigma_min=0.1, sigma_max=100, # VE SDE specific
                    latent_dim=200,  # VAE & GAN
                    n_critic=3, gp_weight=10., # GAN specific
                    bsz=512, train_steps=10000, lr=1e-3, ema_decay=0.999,  # training
                    snapshot_freq=1000, log_freq=500,  # logging
                    )
    add_dict_to_argparser(parser, defaults)
    return parser


def train_diffusion(args, loader, logger):
    # Meta.
    logger.info(f'eps: {args.eps}')
    logger.info(f'T: {args.T}')
    logger.info(f'N: {args.N}')

    # Network, optimizer and diffusion.
    model = UNet().to(device)
    model.train()
    ema = deepcopy(model).to(device).requires_grad_(False)
    ema.eval()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
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

    # Training.
    loss_mean, loss_var, cnt = 0., 0., 0
    for step in range(1, args.train_steps + 1):
        x, y = next(loader)
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = sde.matching_loss_fn(model, x, y)
        loss.sum().mul(1 / x.shape[0]).backward()
        optim.step()
        # EMA update
        for p_ema, p_net in zip(ema.parameters(), model.parameters()):
            if p_net.requires_grad:
                p_ema.copy_(p_net.detach().lerp(p_ema, args.ema_decay))
        # Welford's online update
        cnt += 1
        delta = loss.mean().item() - loss_mean
        loss_mean += delta / cnt
        loss_var += delta * (loss.mean().item() - loss_mean)
        # logging
        if step % args.log_freq == 0 and cnt > 1:
            logger.info(
                f'step [{step:07d}/{args.train_steps:07d}], loss {loss_mean:.3f}(pm {(loss_var / (cnt - 1)) ** 0.5:.3f})')
            loss_mean, loss_var, cnt = 0., 0., 0
        if step % args.snapshot_freq == 0 or step == args.train_steps:
            torch.save(ema, os.path.join(args.rundir, f'network-snapshot-{step // 1000:06d}K.pt'))


def train_vae(args, loader, logger):
    """Reference: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial"""
    # Meta.
    logger.info(f'latent dim: {args.latent_dim}')

    # Network, optimizer and diffusion.
    model = VAE(args.num_classes, args.latent_dim).to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training.
    loss_mean, loss_var, cnt = 0., 0., 0
    for step in range(1, args.train_steps + 1):
        x, y = next(loader)
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = model.loss_fn(x, y)
        loss.sum().mul(1 / x.shape[0]).backward()
        optim.step()
        # Welford's online update
        cnt += 1
        delta = loss.mean().item() - loss_mean
        loss_mean += delta / cnt
        loss_var += delta * (loss.mean().item() - loss_mean)
        # logging
        if step % args.log_freq == 0 and cnt > 1:
            logger.info(
                f'step [{step:07d}/{args.train_steps:07d}], loss {loss_mean:.3f}(pm {(loss_var / (cnt - 1)) ** 0.5:.3f})')
            loss_mean, loss_var, cnt = 0., 0., 0
        if step % args.snapshot_freq == 0 or step == args.train_steps:
            torch.save(model, os.path.join(args.rundir, f'network-snapshot-{step // 1000:06d}K.pt'))


def train_gan(args, loader, logger):
    """Reference: https://github.com/KimRass/Conditional-WGAN-GP"""
    # Meta.
    logger.info(f'latent dim: {args.latent_dim}')
    logger.info(f'n_critic: {args.n_critic}')

    # Network, optimizer and diffusion.
    model = GAN(args.num_classes, args.latent_dim).to(device)
    model.train()
    generator_optim = torch.optim.Adam(model.generator.parameters(), lr=args.lr)
    critic_optim = torch.optim.Adam(model.critic.parameters(), lr=args.lr)

    # Training.
    critic_loss_mean, critic_loss_var, generator_loss_mean, generator_loss_var, cnt = 0., 0., 0., 0., 0
    for step in range(1, args.train_steps + 1):
        x, y = next(loader)
        x, y = x.to(device), y.to(device)

        # critic update
        critic_loss = 0.
        for _ in range(args.n_critic):
            critic_optim.zero_grad()
            adv_loss, gp = model.critic_loss_fn(x, y)
            critic_loss_step = adv_loss + args.gp_weight * gp
            critic_loss_step.sum().mul(1 / x.shape[0]).backward()
            critic_optim.step()
            critic_loss += critic_loss_step

        # generator update
        generator_optim.zero_grad()
        generator_loss = model.generator_loss_fn(x, y)
        generator_loss.sum().mul(1 / x.shape[0]).backward()
        generator_optim.step()

        # Welford's online update
        cnt += 1
        delta = critic_loss.mean().item() - critic_loss_mean # pyright: ignore[reportAttributeAccessIssue]
        critic_loss_mean += delta / cnt
        critic_loss_var += delta * (critic_loss.mean().item() - critic_loss_mean) # pyright: ignore[reportAttributeAccessIssue]
        delta = generator_loss.mean().item() - generator_loss_mean
        generator_loss_mean += delta / cnt
        generator_loss_var += delta * (generator_loss.mean().item() - generator_loss_mean)
        # logging
        if step % args.log_freq == 0 and cnt > 1:
            logger.info(
                f'step [{step:07d}/{args.train_steps:07d}], '
                f'critic loss {critic_loss_mean:.3f}(pm {(critic_loss_var / (cnt - 1)) ** 0.5:.3f}), '
                f'generator loss {generator_loss_mean:.3f}(pm {(generator_loss_var / (cnt - 1)) ** 0.5:.3f})')
            critic_loss_mean, critic_loss_var, generator_loss_mean, generator_loss_var, cnt = 0., 0., 0., 0., 0
        if step % args.snapshot_freq == 0 or step == args.train_steps:
            torch.save(model, os.path.join(args.rundir, f'network-snapshot-{step // 1000:06d}K.pt'))


if __name__ == '__main__':
    # Meta.
    args = create_parser().parse_args()
    desc = f'{args.mode}-n{args.n_train}'
    if args.desc:
        desc += f'-{args.desc}'
    rundir = os.path.join(args.output, desc)
    os.makedirs(rundir, exist_ok=True)
    args.rundir = rundir
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset and DataLoader
    dataset = MNIST(
        root="./dataset", 
        # pixel value [0, 255] -> [0, 1] -> [-1, 1]
        transform=Compose([ToTensor(), lambda x: 2*x-1]),
        target_transform=lambda x: F.one_hot(torch.LongTensor([x]), 10).squeeze(0).float(),
        train=True)
    subset = Subset(dataset, range(args.n_train))
    args.bsz = min(args.n_train, args.bsz)
    loader = DataLoader(subset, batch_size=args.bsz, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.num_workers)
    loader = create_infinite_dataloader(loader)
    args.in_ch = dataset[0][0].shape[0]
    args.out_ch = args.in_ch
    args.img_res = dataset[0][0].shape[-1]
    args.num_classes = dataset[0][1].shape[-1]

    # Logging
    logger = default_logger(os.path.join(args.rundir, 'train.log'))
    logger.info(f'device: {device}')
    logger.info(f'global batch size: {args.bsz}')

    # Training.
    if args.mode in ['follmer', 'linear', 'vesde', 'trig']:
        train_diffusion(args, loader, logger)
    elif args.mode == 'gan':
        train_gan(args, loader, logger)
    elif args.mode == 'vae':
        train_vae(args, loader, logger)
    else:
        raise ValueError(f'unrecognized mode {args.mode}')
