# %%
# !pip install easydict numpy pandas matplotlib seaborn torch scikit-learn ucimlrepo tensorboard

# %%
# %% [markdown]
# # Conditional generative models on UCI wine quality dataset

# %%
import os
import math

from easydict import EasyDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from scipy.stats import t

device = torch.device("cuda")

# %% [markdown]
# ### Hyper-parameters

# %%
scheme = "ode"
sde_name = "SchrodingerFollmer" if scheme == "sde" else "Follmer"

cfg = EasyDict({
    "meta": {
        "desc": f"{scheme}-wine",
    },
    "data": {
        "label_dim": 11,
        "target_dim": 1,
        "train_ratio": 0.75,
        "validation_ratio": 0.15,
        "test_ratio": 0.10,
    },
    "diffusion": {
        "name": sde_name,
    },
    "model": {
        "hidden_dims": [128, 128, 128, 128, 128],
        "dropout": 0.,
        "T": 1000,
    },
    "train": {
        "max_epochs": 50000,
        "eps0": 1e-5,
        "eps1": 1e-5,
        "save_freq": 500,
        "eval_freq": 1,
        "sampling_freq": 2000,
    },
    "optim": {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    },
    "eval": {
    },
    "sampling": {
        "eps0": 1e-3,
        "eps1": 1e-3,
        "T": 1000,
    }
})
cfg

# %% [markdown]
# ## Network

# %%
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = torch.outer(x, (2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
        
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=512):
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
    def __init__(self, in_dim=1, label_dim=None, hidden_dims=[256, 512, 256], dropout=0.):
        super().__init__()
        self.layers = torch.nn.ModuleList([nn.Linear(in_dim, hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], in_dim))
        self.layers.append(nn.Dropout(p=dropout))
        self.emb_layers = torch.nn.ModuleList([PositionalEmbedding(hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.emb_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.label_layers = torch.nn.ModuleList([nn.Linear(label_dim, hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.label_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

    def forward(self, x, t, label=None):
        t_emb = t
        label_emb = label
        for i in range(len(self.layers)-2):
            x = self.layers[i](x)
            t_emb = self.emb_layers[i](t_emb)
            label_emb = self.label_layers[i](label_emb) if label is not None else None
            emb = t_emb + label_emb if label is not None else t_emb
            x = F.silu(x + emb)
        x = self.layers[-2](x)
        x = self.layers[-1](x)
        return x

# %% [markdown]
# ### Score funciton wrapper
# The time variable in unit-time interval $[0, 1]$ is multiplied by a large $T$ when passed to the network for numerical stability.

# %%
def get_score_fn(model, sde, cfg):
    def score_fn(x, t, label):
        score = model(x, t*cfg.model.T, label)
        std = sde.marginal_prob(torch.empty_like(x, device=device), t)[1]
        return score / std[:, None]
    return score_fn

# %% [markdown]
# ## Diffusion

# %%
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
        x = noise if noise is not None else self.prior_sampling(shape)
        dt = (1. - cfg.sampling.eps0 - cfg.sampling.eps1) / cfg.sampling.T
        grid = torch.linspace(1-cfg.sampling.eps1, cfg.sampling.eps0, cfg.sampling.T, device=device)
        for ti in grid[:-1]:
            t = torch.ones((x.shape[0], ), device=x.device) * ti
            drift, diffusion = self.sde(x, t)
            x = x - dt * (drift - 0.5*diffusion[:, None]**2*score_fn(x, t, label))
        return x


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

# %% [markdown]
# ## Prepare dataset, diffusion, model, optimizer, loss funciton and callback plot funciton

# %%
wine_quality = fetch_ucirepo(id=186)
label = wine_quality.data.features.to_numpy().astype(np.float32)
target = wine_quality.data.targets.to_numpy().astype(np.float32)
scalar = StandardScaler()
label = scalar.fit_transform(label)
label_train, label_test, target_train, target_test = train_test_split(label, target, \
                                                                      test_size=1 - cfg.data.train_ratio, shuffle=True)
label_val, label_test, target_val, target_test = train_test_split(label_test, target_test, \
                                                                  test_size=cfg.data.test_ratio/(cfg.data.test_ratio + cfg.data.validation_ratio)) 

target_train = torch.from_numpy(target_train).to(device)
label_train = torch.from_numpy(label_train).to(device)
target_val = torch.from_numpy(target_val).to(device)
label_val = torch.from_numpy(label_val).to(device)
target_test = torch.from_numpy(target_test).to(device)
label_test = torch.from_numpy(label_test).to(device)
sorted_ind = torch.argsort(target_test, dim=0).flatten()
target_test = target_test[sorted_ind]
label_test = label_test[sorted_ind]


sde = SDEs[cfg.diffusion.name]()
model = ToyNet(in_dim=cfg.data.target_dim, label_dim=cfg.data.label_dim, \
               hidden_dims=cfg.model.hidden_dims, dropout=cfg.model.dropout).to(device)
optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas, eps=cfg.optim.eps)
score_fn = get_score_fn(model, sde, cfg)
try:
    os.makedirs(f"logs/{cfg.meta.desc}/ckpt")
except:
    pass
    
def loss_fn(target, label):
    t = torch.rand(target.shape[0], device=device) * (1. - cfg.train.eps0 - cfg.train.eps1) + cfg.train.eps0
    z = torch.randn_like(target, device=device)
    mean, std = sde.marginal_prob(target, t)
    perturbed_data = mean + std[:, None] * z
    score = score_fn(perturbed_data, t, label)
    loss = torch.square(score*std[:, None] + z)
    loss = torch.mean(torch.mean(loss.reshape(loss.shape[0], -1), dim=-1))
    return loss
    
def plot_fn(pred, target, label):
    fig, ax = plt.subplots(1, 1)
    error = target-pred
    sns.kdeplot(x=error.cpu().numpy().flatten(), ax=ax)
    ax.set_xlabel("prediction error")
    return fig

# %% [markdown]
# ## Training

# %%
with SummaryWriter(log_dir=f"logs/{cfg.meta.desc}") as tb_writer:
    for epoch in range(cfg.train.max_epochs):
        # train
        model.train()
        loss = loss_fn(target_train, label_train)
        loss.backward()
        optim.step()
        optim.zero_grad()
        tb_writer.add_scalar("Loss/Train", loss.item(), epoch+1)
        # eval
        if (epoch+1) % cfg.train.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                loss = loss_fn(target_test, label_test)
                tb_writer.add_scalar("Loss/Eval", loss.item(), epoch+1)
        # sampling snapshot
        if (epoch+1) % cfg.train.sampling_freq == 0:
            model.eval()
            with torch.no_grad():
                noise = sde.prior_sampling([target_val.shape[0], cfg.data.target_dim]).to(device)
                pred = sde.sampling(score_fn, label_val, noise=noise)
                fig = plot_fn(pred, target_val, label_val)
                tb_writer.add_figure(f"Sample", fig, epoch+1)
        # save snapshot
        if (epoch+1) % cfg.train.save_freq == 0:
            torch.save(dict(model=model.state_dict(), optim=optim.state_dict(), epoch=epoch), \
                       f"logs/{cfg.meta.desc}/ckpt/{epoch+1}.pth")


# %%
np.random.seed(42)
model.load_state_dict(torch.load(f"./logs/{cfg.meta.desc}/last.pth")['model'])
with torch.no_grad():
    model.eval()
    noise = sde.prior_sampling([target_test.shape[0], cfg.data.target_dim]).to(device)
    pred_test = sde.sampling(score_fn, label_test, noise=noise)
    fig = plot_fn(pred_test, target_test, label_test)
    plt.show()

# %%
alpha = 0.1
residuals = pred_test - target_test
residual_std = torch.std(residuals)
se = residual_std * math.sqrt(1 + 1/len(pred_test))
t_value = t.ppf(1 - alpha / 2, df=len(pred_test) - label_test.shape[1])
lower_bound = pred_test - t_value * se
upper_bound = pred_test + t_value * se
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.vlines(np.arange(len(pred_test)), ymin=lower_bound.cpu().numpy().flatten(), ymax=upper_bound.cpu().numpy().flatten(), \
           colors="dimgrey", linestyles='solid', linewidth=0.4, alpha=0.5)
plt.hlines(lower_bound.cpu().numpy().flatten(), xmin=np.arange(len(pred_test)) - 2.5, xmax=np.arange(len(pred_test)) + 2.5, \
           colors="dimgrey", linestyles='solid')
plt.hlines(upper_bound.cpu().numpy().flatten(), xmin=np.arange(len(pred_test)) - 2.5, xmax=np.arange(len(pred_test)) + 2.5, \
           colors="dimgrey", linestyles='solid')
plt.scatter(np.arange(len(pred_test)), target_test.cpu().numpy().flatten(), s=0.5, color='black', label="truth")
ax.set_xlabel("Case number")
ax.set_ylabel("Wine quality")
plt.grid(linestyle=':')
plt.savefig(f"logs/{cfg.meta.desc}/wine_prediction_interval.pdf", dpi=500, bbox_inches="tight", pad_inches=0.1)

# %%
torch.sum((lower_bound <= target_test) & (target_test <= upper_bound)) / len(pred_test)

# %%
with torch.no_grad():
    model.eval()
    noise = sde.prior_sampling([target_train.shape[0], cfg.data.target_dim]).to(device)
    pred_train = sde.sampling(score_fn, label_train, noise=noise)
    noise = sde.prior_sampling([target_test.shape[0], cfg.data.target_dim]).to(device)
    pred_test = sde.sampling(score_fn, label_test, noise=noise)

lst = []
ntest = pred_test.shape[0]
lst += [(pred_test.cpu().numpy().flatten()[i], "prediction") for i in range(ntest)]
lst += [(target_test.cpu().numpy().flatten()[i], "truth") for i in range(ntest)]
df = pd.DataFrame(lst, columns=["data", "type"])
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.kdeplot(x=pred_test.cpu().numpy().flatten(), ax=ax, label="prediction", bw_adjust=3, linewidth=2, gridsize=50, c='black', marker='v')
sns.kdeplot(x=target_test.cpu().numpy().flatten(), ax=ax, label="truth", bw_adjust=3, linewidth=2, gridsize=50, c='dimgrey', marker='d')
# sns.kdeplot(df, x="data", hue="type", bw_adjust=3, palette="Greys", ax=ax, legend=True)
ax.set_xlabel("Wine quality")
ax.grid(linestyle=':')
ax.legend()
plt.savefig(f"logs/{cfg.meta.desc}/wine_kde.pdf", bbox_inches="tight", pad_inches=0.1)



