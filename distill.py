# %%
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:0")

ntrain = 2000
ntest = 100
hidden_dims = [256, 512]
seed = 42
eps0 = 1e-3
eps1 = 1e-3
T = 1000
traj_T = 50
nepochs = 50000
lr = 1e-4
bsz = 500
nrepeat = 5
eval_freq = 1
sampling_freq = 1000
save_freq = 5000
desc = f"distill-os"

np.random.seed(seed)
torch.manual_seed(seed)

def solution(x):
    return 1 - 6*x + 36*x**2 - 53*x**3 + 22*x**5 + 0.5*torch.sin(6*torch.pi*x)

def add_noise(y):
    return y + math.sqrt(0.05) * torch.randn_like(y, device=y.device)

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
    def __init__(self, in_dim=1, cond_dim=None, hidden_dims=[128, 256, 128]):
        super().__init__()
        self.layers = torch.nn.ModuleList([nn.Linear(in_dim, hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], in_dim))
        self.emb_layers = torch.nn.ModuleList([PositionalEmbedding(hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.emb_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.cond_layers = torch.nn.ModuleList([nn.Linear(cond_dim, hidden_dims[0])])
        for i in range(len(hidden_dims) - 1):
            self.cond_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

    def forward(self, x, t, cond=None):
        t_emb = t
        cond_emb = cond
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            t_emb = self.emb_layers[i](t_emb)
            cond_emb = self.cond_layers[i](cond_emb) if cond is not None else None
            emb = t_emb + cond_emb if cond is not None else t_emb
            x = F.silu(x + emb)
        x = self.layers[-1](x)
        return x

def get_score_fn(model, sde):
    def score_fn(x, t, cond):
        score = model(x, t*T, cond)
        std = sde.marginal_prob(torch.empty_like(x, device=device), t)[1]
        return score / std[:, None]
    return score_fn

def get_traj_fn(model):
    def traj_fn(x, t, cond):
        return t[:, None]*x + (1-t[:, None])*model(x, t*traj_T, cond)
    return traj_fn

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
    
    def velocity(self, score_fn, x, t, cond):
        drift, diffusion = self.sde(x, t)
        return (drift - 0.5*diffusion[:, None]**2*score_fn(x, t, cond))
    
    def euler_step(self, score_fn, x, t, dt, cond):
        return x + dt[:, None] * self.velocity(score_fn, x, t, cond)
    
    def heun_step(self, score_fn, x, t, dt, cond):
        velocity1 = self.velocity(score_fn, x, t, cond)
        x_euler = x + dt[:, None] * velocity1
        velocity2 = self.velocity(score_fn, x_euler, t+dt, cond)
        return x + dt[:, None] * (velocity1 + velocity2) / 2
    
    def solve(self, score_fn, x, cond, t=None, u=None, T=T, method="euler"):
        if t is None:
            t = torch.ones(x.shape[0], device=device) * (1-eps1)
        if u is None:
            u = torch.ones(x.shape[0], device=device) * eps0
        dt = (t - u) / T
        for i in range(T-1):
            ti = t - i * dt
            if method == "euler":
                x = self.euler_step(score_fn, x, ti, -dt, cond)
            elif method == "heun":
                x = self.heun_step(score_fn, x, ti, -dt, cond)
        return x


class SchrodingerFollmer:
    def __init__(self):
        pass

    def sde(self, x, t):
        drift = x / (t[:, None] - 1)
        diffusion = torch.ones_like(t, device=device)
        return drift, diffusion
    
    def rev_sde(self, score_fn, x, t, cond):
        drift, diffusion = self.sde(x, t)
        return drift - diffusion[:, None]**2*score_fn(x, t, cond), diffusion

    def marginal_prob(self, x, t):
        mean = (1 - t[:, None]) * x
        std = torch.sqrt(t*(1-t))
        return mean, std

    def prior_sampling(self, shape):
        return torch.zeros(shape)
    
    def em_step(self, score_fn, x, t, dt, cond):
        drift, diffusion = self.rev_sde(score_fn, x, t, cond)
        z = torch.randn_like(x, device=device)
        return x + drift*dt[:, None] + diffusion[:, None]*torch.sqrt(torch.abs(dt[:, None]))*z

    def solve(self, score_fn, x, cond, t=None, u=None, T=T, method="em"):
        if t is None:
            t = torch.ones(x.shape[0], device=device) * (1-eps1)
        if u is None:
            u = torch.ones(x.shape[0], device=device) * eps0
        dt = (t - u) / T
        for i in range(T-1):
            ti = t - i * dt
            if method == "em":
                x = self.em_step(score_fn, x, ti, -dt, cond)
        return x

sde = Follmer()
score_model = ToyNet(in_dim=1, cond_dim=1, \
               hidden_dims=hidden_dims).to(device)
score_model.load_state_dict(torch.load("logs/ode-os/last.pth", device)["model"])
score_fn = get_score_fn(score_model, sde)
traj_model = ToyNet(in_dim=1, cond_dim=1, \
               hidden_dims=hidden_dims).to(device)
optim = torch.optim.Adam(traj_model.parameters(), lr)
traj_fn = get_traj_fn(traj_model)
    
def plot_fn(pred, truth, cond):
    pred = pred.detach().cpu().numpy().flatten()
    truth = truth.detach().cpu().numpy().flatten()
    cond = cond.detach().cpu().numpy().flatten()
    lst = []
    lst = [(cond[i], truth[i], "truth") for i in range(cond.shape[0])]
    lst += [(cond[i], pred[i], "prediction") for i in range(cond.shape[0])]
    data = pd.DataFrame(lst, columns=["x", "y", "type"])
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data, x="x", y="y", hue="type", ax=ax)
    return fig
# %%
try:
    os.makedirs(f"logs/{desc}/ckpt")
except:
    pass

with SummaryWriter(log_dir=f"logs/{desc}") as tb_writer:
    for epoch in range(nepochs):
        # train
        traj_model.train()
        cond = torch.rand((bsz//nrepeat, 1), device=device).repeat([nrepeat, 1])
        data = add_noise(solution(cond))
        data_t = torch.empty((bsz, traj_T), device=device)
        data_t[:, 0:1] = data
        t = torch.ones(data_t.shape[0], device=device) * eps0
        u = torch.ones(data_t.shape[0], device=device) * (1-eps1)
        dt = (u - t)/traj_T
        with torch.no_grad():
            for i in range(traj_T-1):
                ti = t + i*dt
                data_t[:, i+1:i+2] = sde.heun_step(score_fn, data_t[:, i:i+1], ti, dt, cond)
        noise = data_t[:, -1][:, None]
        choices = torch.randint(0, traj_T, (bsz,), device=device)
        target = data_t.gather(1, choices[:, None])
        pred = traj_fn(noise, choices/traj_T, cond)
        loss = torch.mean(torch.square(pred - target))
        # choices = torch.zeros((bsz,), device=device).long()
        # target = data_t[:, 0][:, None]
        # pred = traj_fn(noise, choices/traj_T, cond)
        # loss = torch.mean(torch.square(pred - target))
        loss.backward()
        optim.step()
        optim.zero_grad()
        tb_writer.add_scalar("Loss/Train", loss.item(), epoch+1)
        # sampling snapshot
        if (epoch+1) % sampling_freq == 0:
            traj_model.eval()
            with torch.no_grad():
                noise = sde.prior_sampling([bsz, 1]).to(device)
                pred = traj_fn(noise, torch.zeros((bsz, ), device=device), cond=cond)
                target = data_t[:, 0]
                fig = plot_fn(pred, target, cond)
                tb_writer.add_figure(f"Prediction", fig, epoch+1)
        # save snapshot
        if (epoch+1) % save_freq == 0:
            torch.save(dict(model=traj_model.state_dict(), optim=optim.state_dict(), epoch=epoch), \
                       f"logs/{desc}/ckpt/{epoch+1}.pth")
# %%
def plot_fn(pred, truth, cond):
    pred = pred.detach().cpu().numpy().flatten()
    truth = truth.detach().cpu().numpy().flatten()
    cond = cond.detach().cpu().numpy().flatten()
    lst = []
    lst = [(cond[i], truth[i], "truth") for i in range(cond.shape[0])]
    lst += [(cond[i], pred[i], "prediction") for i in range(cond.shape[0])]
    data = pd.DataFrame(lst, columns=["x", "y", "type"])
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data, x="x", y="y", hue="type", ax=ax)
    ax.legend(title=None)
    return fig

nrepeat = 5
ntest = 25
x_test = torch.linspace(0, 1, ntest, device=device)[:, None].repeat([nrepeat, 1])
y_test = add_noise(solution(x_test))

sde = Follmer()
score_model = ToyNet(in_dim=1, cond_dim=1, \
               hidden_dims=hidden_dims).to(device)
score_fn = get_score_fn(score_model, sde)
score_model.load_state_dict(torch.load("logs/ode-os/last.pth", device)["model"])
score_model.eval()
with torch.no_grad():
    noise = sde.prior_sampling([x_test.shape[0], 1]).to(device)
    pred = sde.solve(score_fn, noise, cond=x_test)
    pred_ode = torch.from_numpy(pred.cpu().numpy())

sde = SchrodingerFollmer()
score_model = ToyNet(in_dim=1, cond_dim=1, \
               hidden_dims=hidden_dims).to(device)
score_fn = get_score_fn(score_model, sde)
score_model.load_state_dict(torch.load("logs/sde-os/last.pth", device)["model"])
score_model.eval()
with torch.no_grad():
    noise = sde.prior_sampling([x_test.shape[0], 1]).to(device)
    pred = sde.solve(score_fn, noise, cond=x_test)
    pred_sde = torch.from_numpy(pred.cpu().numpy())

sde = Follmer()
score_fn = get_score_fn(score_model, sde)
traj_model = ToyNet(in_dim=1, cond_dim=1, \
               hidden_dims=hidden_dims).to(device)
traj_fn = get_traj_fn(traj_model)
traj_model.load_state_dict(torch.load("logs/distill-os/last.pth", device)["model"])
with torch.no_grad():
    noise = sde.prior_sampling([x_test.shape[0], 1]).to(device)
    pred_traj = traj_fn(noise, torch.zeros((x_test.shape[0], ), device=device), cond=x_test)

pred_ode = pred_ode.detach().cpu().numpy().flatten()
pred_sde = pred_sde.detach().cpu().numpy().flatten()
pred_traj = pred_traj.detach().cpu().numpy().flatten()
target = y_test.detach().cpu().numpy().flatten()
cond = x_test.detach().cpu().numpy().flatten()
lst = []
lst = [(cond[i], target[i], "observations") for i in range(cond.shape[0])]
# lst += [(cond[i], pred_ode[i], "SDE") for i in range(cond.shape[0])]
lst += [(cond[i], pred_sde[i], "ODE") for i in range(cond.shape[0])]
lst += [(cond[i], pred_traj[i], "Characteristic") for i in range(cond.shape[0])]
data = pd.DataFrame(lst, columns=["x", "y", "type"])
fig, ax = plt.subplots(1, 1, figsize=(8, 2.5))
sns.scatterplot(data[data["type"] == "observations"], x="x", y="y", hue="type", style="type", ax=ax, s=18, markers=["v"], palette="Greys")
sns.lineplot(data[data["type"] != "observations"], x="x", y="y", hue="type", style="type", ax=ax, markers=True, palette="Greys")
ax.legend(title=None)
ax.grid(linestyle=":")
plt.savefig(f"asset/os-comparison.pdf", dpi=300, bbox_inches="tight", pad_inches=0.)
# %%
