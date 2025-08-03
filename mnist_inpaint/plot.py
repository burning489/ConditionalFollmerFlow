import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import argparse
import torch

from diffusion import Follmer, Linear, VESDE, Trigonometric
from misc import add_dict_to_argparser, mask_fn


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(
        device=0, ckpt="", num_classes=10, n_per_class=5, mask_mode=1, mode="follmer",
        img_res=28, in_ch=1, out_ch=1,
        latent_dim=50,  # VAE/GAN
        eps=1e-3, T=0.99999, N=100,  # diffusion
        ode_solver="euler",  # flow-based specific
        sigma_min=0.02, sigma_max=100, sde_solver="euler-maruyama",  # VE SDE specific
    )
    add_dict_to_argparser(parser, defaults)
    return parser

def unmask_ground_truch(mask_mode, reference, x):
    if mask_mode == 1:
        x[..., :14, :] = reference[..., :14, :]
        x[..., :, 14:] = reference[..., :, 14:]
    elif mask_mode == 2:
        x[..., :, 14:] = reference[..., :, 14:]
    elif mask_mode == 3:
        x[..., 14:, 14:] = reference[..., 14:, 14:]
    else:
        raise ValueError(f'unrecognized mask mode {mask_mode}')
    return x

def plot_diffusion(args, x_ref, y):
    # Network and diffusion.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()
    if args.mode == "follmer":
        sde = Follmer(args)
    elif args.mode == "linear":
        sde = Linear(args)
    elif args.mode == 'trig':
        sde = Trigonometric(args)
    elif args.mode == "vesde":
        sde = VESDE(args)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")

    y = y.to(device)
    noise = sde.prior_sampling(shape=[y.shape[0], args.in_ch, args.img_res, args.img_res], device=device)
    x = sde.solve(model, noise, y)
    # x = unmask_ground_truch(args.mask_mode, x_ref, x)
    x = (x+1)/2
    return make_grid(x.clamp(0., 1.), nrow=args.n_per_class)

def plot_vae(args, x_ref, y):
    # Network.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    y = y.to(device)
    with torch.no_grad():
        noise = torch.randn((y.shape[0], args.latent_dim), device=device)
        x = model.decoder(noise, y)
        # x = unmask_ground_truch(args.mask_mode, x_ref, x)
        x = (x+1)/2
    return make_grid(x.clamp(0., 1.), nrow=args.n_per_class)

def plot_gan(args, x_ref, y):
    # Network.
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    y = y.to(device)
    with torch.no_grad():
        noise = torch.randn((y.shape[0], args.latent_dim), device=device)
        x = model.generator(noise, y)
        # x = unmask_ground_truch(args.mask_mode, x_ref, x)
        x = (x+1)/2
    return make_grid(x.clamp(0., 1.), nrow=args.n_per_class)


if __name__ == "__main__":
    args = create_parser().parse_args()
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt)
    class_idx = list(range(10))
    samples = torch.empty(10, 1, 28, 28, device=device)

    dataset = MNIST(root="./dataset", transform=ToTensor(), train=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for data, labels in loader:
        label = labels.item()
        if label in class_idx:
            samples[label, ...] = data
            class_idx.remove(int(label))
        if not class_idx:
            break

    samples_grid = make_grid(samples, nrow=1)
    samples_rgb = samples.repeat(1, 3, 1, 1)
    if args.mask_mode == 1:
        samples_rgb[:, 0, :14, :] = 1
        samples_rgb[:, 1:, :14, :] = 0
        samples_rgb[:, 0, :, 14:] = 1
        samples_rgb[:, 1:, :, 14:] = 0
    elif args.mask_mode == 2:
        samples_rgb[:, 0, :, 14:] = 1
        samples_rgb[:, 1:, :, 14:] = 0
    elif args.mask_mode == 3:
        samples_rgb[:, 0, 14:, 14:] = 1
        samples_rgb[:, 1:, 14:, 14:] = 0
    else:
        raise ValueError(f"unrecognized mask mode {args.mask_mode}")
    masked_grid = make_grid(samples_rgb, nrow=1)

    x = samples.repeat(args.n_per_class, 1, 1, 1)
    y = mask_fn(args.mask_mode)(x)
    y = y.reshape(args.n_per_class, args.num_classes, *y.shape[1:]).transpose(1, 0).reshape(-1, *y.shape[1:])
    x, y = 2*x-1, 2*y-1

    if args.mode in ["follmer", "linear", "vesde", 'trig']:
        generated_grid = plot_diffusion(args, x, y)
    elif args.mode == 'vae':
        generated_grid = plot_vae(args, x, y)
    elif args.mode == 'gan':
        generated_grid = plot_gan(args, x, y)
    else:
        raise ValueError(f"unsupported eval mode {args.mode}")

    grid = torch.concat([samples_grid, masked_grid, generated_grid], dim=2)
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(os.path.join(rundir, f"samples.png"), dpi=300, bbox_inches="tight", pad_inches=0.0)
