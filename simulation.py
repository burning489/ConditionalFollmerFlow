# %%
# !pip install easydict numpy matplotlib seaborn torch scikit-learn tensorboard

# %%
import os
import math

from easydict import EasyDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device(f"cuda:0")

model_id = 2
scheme = "ode"
sde_name = "SchrodingerFollmer" if scheme == "sde" else "Follmer"
LABEL_DIMs = {1: 5, 2: 5, 3: 1}

def m1(x):
    y = x[:, 0]**2 + np.exp(x[:, 1] + x[:, 2]/3) + np.sin(x[:, 3] + x[:, 4]) + np.random.randn(x.shape[0])
    return y[:, None]

def m2(x):
    z = np.random.randn(x.shape[0])
    y = x[:, 0]**2 + np.exp(x[:, 1] + x[:, 2]/3) + x[:, 3] - x[:, 4] + (1 + x[:, 1]**2 + x[:, 4]**2) / 2 * z
    return y[:, None]

def m3(x):
    choices = np.random.rand(x.shape[0], 1)
    y = 0.25*np.random.randn(x.shape[0], 1) + x * np.where(choices>0.5, 1, -1)
    return y

MODELs = {1: m1, 2: m2, 3: m3}

cfg = EasyDict({
    "meta": {
        "desc": f"{scheme}-simulation{model_id}",
        "seed": 42,
    },
    "data": {
        "model": model_id,
        "label_dim": LABEL_DIMs[model_id],
        "target_dim": 1,
        "nsample": 60000,
        "test_ratio": 0.1
    },
    "diffusion": {
        "name": sde_name
    },
    "model": {
        "hidden_dims": [256, 512],
        "T": 1000,
    },
    "train": {
        "max_epochs": 100000,
        "eps0": 1e-3,
        "eps1": 1e-3,
        "save_freq": 1000,
        "eval_freq": 1,
        "sampling_freq": 1000,
    },
    "optim": {
        "lr": 5e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    },
    "eval": {
    },
    "sampling": {
        "bsz": 5000,
        "eps0": 1e-3,
        "eps1": 1e-3,
        "T": 1000,
    }
})

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = torch.outer(x, (2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
        
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        freqs = torch.linspace(0, 0.5, num_channels//2)
        freqs = (1 / self.max_positions) ** freqs
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        x = torch.outer(x, self.freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
        
class ToyNet(nn.Module):
    def __init__(self, in_dim=1, label_dim=None, hidden_dims=[128, 256, 128]):
        super().__init__()
        self.layers = torch.nn.ModuleList([nn.Linear(in_dim, hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], in_dim))
        self.emb_layers = torch.nn.ModuleList([PositionalEmbedding(hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.emb_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.label_layers = torch.nn.ModuleList([nn.Linear(label_dim, hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.label_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

    def forward(self, x, t, label=None):
        t_emb = t
        label_emb = label
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            t_emb = self.emb_layers[i](t_emb)
            label_emb = self.label_layers[i](label_emb) if label is not None else None
            emb = t_emb + label_emb if label is not None else t_emb
            x = F.silu(x + emb)
        x = self.layers[-1](x)
        return x

def get_score_fn(model, sde, cfg):
    def score_fn(x, t, label):
        score = model(x, t*cfg.model.T, label)
        std = sde.marginal_prob(torch.empty_like(x, device=device), t)[1]
        return score / std[:, None]
    return score_fn


class Follmer:
    def __init__(self):
        pass

    def sde(self, x, t):
        drift = x / (t[:, None] - 1)
        diffusion = torch.sqrt(2/(1-t))
        return drift, diffusion

    def marginal_prob(self, x, t):
        mean = (1 - t[:, None]) * x
        std = torch.sqrt(t*(2-t))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(shape)

    def sampling(self, score_fn, label=None, shape=None, noise=None):
        xt = noise if noise is not None else self.prior_sampling(shape)
        dt = (1. - cfg.sampling.eps0 - cfg.sampling.eps1) / cfg.sampling.T
        grid = torch.linspace(1-cfg.sampling.eps1, cfg.sampling.eps0, cfg.sampling.T, device=device)
        for ti in grid[:-1]:
            t = torch.ones((xt.shape[0], ), device=xt.device) * ti
            drift, diffusion = self.sde(xt, t)
            xt = xt - dt * (drift - 0.5*diffusion[:, None]**2*score_fn(xt, t, label))
        return xt


class SchrodingerFollmer:
    def __init__(self):
        pass

    def sde(self, x, t):
        drift = x / (t[:, None] - 1)
        diffusion = torch.ones_like(t, device=device)
        return drift, diffusion

    def marginal_prob(self, x, t):
        mean = (1 - t[:, None]) * x
        std = torch.sqrt(t*(1-t))
        return mean, std

    def prior_sampling(self, shape):
        return torch.zeros(shape)

    def sampling(self, score_fn, label=None, shape=None, noise=None):
        x = noise if noise is not None else self.prior_sampling(shape)
        dt = (1. - cfg.sampling.eps0 - cfg.sampling.eps1) / cfg.sampling.T
        grid = torch.linspace(1-cfg.sampling.eps1, cfg.sampling.eps0, cfg.sampling.T, device=device)
        for ti in grid[:-1]:
            t = torch.ones((x.shape[0], ), device=x.device) * ti
            drift, diffusion = self.sde(x, t)
            diffusion = diffusion[:, None]
            z = torch.randn_like(x, device=x.device)
            x = x - dt * (drift - diffusion**2*score_fn(x, t, label)) + diffusion*math.sqrt(dt)*z
        return x

SDEs = {
    "SchrodingerFollmer": SchrodingerFollmer,
    "Follmer": Follmer
}


target_fn = MODELs[cfg.data.model]
rng = np.random.RandomState(cfg.meta.seed)
x = rng.randn(cfg.data.nsample, cfg.data.label_dim)
y = target_fn(x)
scaler = StandardScaler()
y = scaler.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cfg.data.test_ratio)
x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)
x_eval = np.tile(np.random.randn(1, cfg.data.label_dim), (cfg.sampling.bsz, 1)).astype(np.float32)
y_eval = target_fn(x_eval).astype(np.float32)
x_eval = torch.from_numpy(x_eval).to(device)
y_eval = torch.from_numpy(y_eval).to(device)


sde = SDEs[cfg.diffusion.name]()
model = ToyNet(in_dim=cfg.data.target_dim, label_dim=cfg.data.label_dim, \
               hidden_dims=cfg.model.hidden_dims).to(device)
optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas, eps=cfg.optim.eps)
score_fn = get_score_fn(model, sde, cfg)

try:
    os.makedirs(f"logs/{cfg.meta.desc}/ckpt")
except:
    pass
    
def loss_fn(y, x):
    t = torch.rand(y.shape[0], device=device) * (1. - cfg.train.eps0 - cfg.train.eps1) + cfg.train.eps0
    z = torch.randn_like(y, device=device)
    mean, std = sde.marginal_prob(y, t)
    perturbed_x = mean + std[:, None] * z
    score = score_fn(perturbed_x, t, x)
    loss = torch.square(score*std[:, None] + z)
    loss = torch.mean(torch.mean(loss.reshape(loss.shape[0], -1), dim=-1))
    return loss
    
def plot_fn(y_pred, y_true, x):
    fig, ax = plt.subplots(1, 1)
    sns.kdeplot(x=y_true.cpu().numpy().flatten(), ax=ax, label="truth")
    sns.kdeplot(x=y_pred.cpu().numpy().flatten(), ax=ax, label="prediction")
    plt.legend()
    return fig

# %%
with SummaryWriter(log_dir=f"logs/{cfg.meta.desc}") as tb_writer:
    for epoch in range(cfg.train.max_epochs):
        # train
        model.train()
        loss = loss_fn(y_train, x_train)
        loss.backward()
        optim.step()
        optim.zero_grad()
        tb_writer.add_scalar("Loss/Train", loss.item(), epoch+1)
        # eval
        if (epoch+1) % cfg.train.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                loss = loss_fn(y_test, x_test)
                tb_writer.add_scalar("Loss/Eval", loss.item(), epoch+1)
        # sampling snapshot
        if (epoch+1) % cfg.train.sampling_freq == 0:
            model.eval()
            with torch.no_grad():
                noise = sde.prior_sampling([cfg.sampling.bsz, cfg.data.target_dim]).to(device)
                pred = sde.sampling(score_fn, x_eval, noise=noise)
                pred = torch.from_numpy(scaler.inverse_transform(pred.cpu().numpy()))
                fig = plot_fn(pred, y_eval, x_eval)
                tb_writer.add_figure(f"Density", fig, epoch+1)
        # save snapshot
        if (epoch+1) % cfg.train.save_freq == 0:
            torch.save(dict(model=model.state_dict(), optim=optim.state_dict(), epoch=epoch), \
                       f"logs/{cfg.meta.desc}/ckpt/{epoch+1}.pth")

# %%       
ntest = 50
n_repeat = 5
seed = 42
n_mc = 5000
x_dim = LABEL_DIMs[model_id]

# sde = SDEs[cfg.diffusion.name]()
# model = ToyNet(in_dim=cfg.data.target_dim, label_dim=cfg.data.label_dim, \
#                hidden_dims=cfg.model.hidden_dims).to(device)
# score_fn = get_score_fn(model, sde, cfg)
# model.load_state_dict(torch.load(f"logs/{cfg.meta.desc}/ckpt/100000.pth", device)["model"])

mean_mse_vec = np.empty(n_repeat)
std_mse_vec = np.empty(n_repeat)
with torch.no_grad():
    model.eval()
    for n in range(n_repeat):
        x_test = np.tile(rng.randn(ntest, x_dim), (n_mc, 1)).astype(np.float32)
        y_test = target_fn(x_test).reshape(n_mc, ntest)
        x_test = torch.from_numpy(x_test).to(device)
        noise = sde.prior_sampling([n_mc*ntest, 1]).to(device)
        pred = sde.sampling(score_fn, x_test, noise=noise).cpu().numpy()
        pred = scaler.inverse_transform(pred).reshape(n_mc, ntest)
        mean_mse_vec[n] = np.mean((np.mean(pred, axis=0) - np.mean(y_test, axis=0))**2)
        std_mse_vec[n] = np.mean((np.std(pred, axis=0) - np.std(y_test, axis=0))**2)
print(f"mean: {mean_mse_vec.mean():4.3f} pm {mean_mse_vec.std():4.3f} \
    standard deviation: {std_mse_vec.mean():4.3f} pm {std_mse_vec.std():4.3f}")


