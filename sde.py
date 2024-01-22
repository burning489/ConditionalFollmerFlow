import math
import torch


class Follmer:
    def __init__(self):
        pass

    def sde(self, x, t):
        drift = x / (t[:, None, None, None] - 1)
        diffusion = torch.sqrt(2/(1-t))
        return drift, diffusion

    def marginal_prob(self, x, t):
        mean = (1 - t[:, None, None, None]) * x
        std = torch.sqrt(t*(2-t))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(shape)

    def sampling(self, score_fn, noise, label=None, eps0=1e-3, eps1=1e-3, T=1000, mask=None, reference=None):
        x = noise
        dt = (1. - eps0 - eps1) / T
        grid = torch.linspace(1-eps1, eps0, T, device=noise.device)
        for ti in grid[:-1]:
            t = torch.ones((x.shape[0], ), device=x.device) * ti
            drift, diffusion = self.sde(x, t)
            x = x - dt * (drift - 0.5*diffusion[:, None, None, None]**2*score_fn(x, t, label))
            if mask is not None:
                mean, std = self.marginal_prob(reference, t)
                masked_data = mean + torch.randn_like(x, device=x.device) * std[:, None, None, None]
                x = x * (1. - mask) + masked_data * mask
        return x


class SchrodingerFollmer:
    def __init__(self):
        pass

    def sde(self, x, t):
        drift = x / (t[:, None, None, None] - 1)
        diffusion = torch.ones_like(t, device=x.device)
        return drift, diffusion

    def marginal_prob(self, x, t):
        mean = (1 - t[:, None, None, None]) * x
        std = torch.sqrt(t*(1-t))
        return mean, std

    def prior_sampling(self, shape):
        return torch.zeros(shape)

    def sampling(self, score_fn, noise, label=None, eps0=1e-3, eps1=1e-3, T=1000, mask=None, reference=None):
        x = noise
        dt = (1. - eps0 - eps1) / T
        grid = torch.linspace(1-eps1, eps0, T, device=x.device)
        for ti in grid[:-1]:
            t = torch.ones((x.shape[0], ), device=x.device) * ti
            drift, diffusion = self.sde(x, t)
            diffusion = diffusion[:, None, None, None]
            z = torch.randn_like(x, device=x.device)
            x = x - dt * (drift - diffusion**2*score_fn(x, t, label)) + diffusion*math.sqrt(dt)*z
            if mask is not None:
                mean, std = self.marginal_prob(reference, t)
                masked_data = mean + torch.randn_like(x, device=x.device) * std[:, None, None, None]
                x = x * (1. - mask) + masked_data * mask
        return x