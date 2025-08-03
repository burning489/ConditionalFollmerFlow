import os
import argparse

import numpy as np
from scipy import linalg
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


from misc import add_dict_to_argparser, default_logger


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(
        device=0, mode="ref", # meta
        bsz=128, num_workers=1, # loader
        ref="ckpt/ref.npz", # where to store/load reference stats
        ckpt="", # where to load generated samples
        )
    add_dict_to_argparser(parser, defaults)
    return parser

def create_dataloader(dataset, bsz, num_workers=1):
    loader = DataLoader(dataset, batch_size=bsz, num_workers=num_workers)
    yield from loader

def calc_stats(args, logger):
    feature_dim = 784
    if args.mode == 'ref':
        dataset = MNIST(root="./dataset", transform=ToTensor(), train=True)
    elif args.mode == 'gen':
        x = torch.from_numpy(np.load(args.ckpt)).float() / 255.
        dataset = TensorDataset(x)
    else:
        raise ValueError(f'unrecognized mode {args.mode}')
    loader = create_dataloader(dataset, args.bsz, args.num_workers)
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    cnt = 0
    for batch in loader:
        image = batch[0]
        features = image.to(device).view(image.shape[0], -1)
        mu += features.sum(0)
        sigma += features.T @ features
        cnt += image.shape[0]
        logger.info(f'image [{cnt}/{len(dataset)}]')

    mu /= len(dataset)
    sigma -= mu.ger(mu) * len(dataset)
    sigma /= len(dataset) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

def calculate_fid_from_stats(mu1, mu2, sigma1, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


if __name__ == '__main__':
    args = create_parser().parse_args()
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    rundir = os.path.dirname(args.ckpt) if args.ckpt else os.path.dirname(args.ref)

    # Logging
    logger = default_logger(os.path.join(rundir, 'fid.log'))
    logger.info(f'device: {device}')
    logger.info(f'mode: {args.mode}')
    logger.info(f'global batch size: {args.bsz}')
    logger.info(f'num_workers: {args.num_workers}')

    if args.mode == 'ref':
        mu, sigma = calc_stats(args, logger)
        os.makedirs(os.path.dirname(args.ref), exist_ok=True)
        np.savez(args.ref, mu=mu, sigma=sigma)
    elif args.mode == 'gen':
        mu, sigma = calc_stats(args, logger)
        np.savez(os.path.join(rundir, 'stats.npz'), mu=mu, sigma=sigma)
    elif args.mode == 'fid':
        ref = np.load(args.ref)
        stat = np.load(os.path.join(rundir, 'stats.npz'))
        fid = calculate_fid_from_stats(ref['mu'], stat['mu'], ref['sigma'], stat['sigma'])
        logger.info(f"FID distance: {fid:.2f}")