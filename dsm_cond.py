import os
import importlib
import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from network import UNet
from ema import ExponentialMovingAverage
from sde import Follmer

dist.init_process_group(backend="nccl", init_method="env://")
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
cfg = importlib.import_module("config.ode_mnist_cond").cfg

print("Init Module")
train_set = MNIST("./data", train=True, transform=ToTensor())
test_set = MNIST("./data", train=False, transform=ToTensor())
train_sampler = DistributedSampler(train_set)
test_sampler = DistributedSampler(test_set)
train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers, sampler=train_sampler)
test_loader = DataLoader(test_set, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers, sampler=test_sampler)
model = UNet(**cfg.model).to(device)
optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas, eps=cfg.optim.eps)
sde = Follmer()
model = DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False)
ema = ExponentialMovingAverage(model.parameters(), cfg.train.ema_decay)
state = {"model": model, "ema": ema, "optim": optim, "epoch": 0}

def get_score_fn():
    def score_fn(x, t, label):
        # exaggerate temporal variable by T for numerical stability
        score = model(x, t*cfg.sampling.T, label)
        # scale output
        std = sde.marginal_prob(torch.empty_like(x, device=x.device), t)[1]
        return score / std[:, None, None, None]
    return score_fn
ema = ExponentialMovingAverage(model.parameters(), cfg.train.ema_decay)

score_fn = get_score_fn()

def loss_fn(data, label):
    t = torch.rand(data.shape[0], device=data.device) * (1. - cfg.train.eps0 - cfg.train.eps1) + cfg.train.eps0
    z = torch.randn_like(data, device=data.device)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t, label)
    loss = torch.square(score*std[:, None, None, None] + z)
    loss = torch.mean(torch.mean(loss.reshape(loss.shape[0], -1), dim=-1))
    return loss

tb_writer = SummaryWriter(log_dir=f"logs/{cfg.meta.desc}") if dist.get_rank() == 0 else None

try:
    os.makedirs(f"logs/{cfg.meta.desc}/ckpt")
except:
    pass

print("training")
for epoch in range(cfg.train.max_epochs):
    state["epoch"] = epoch
    # train
    model.train()
    loss_epoch = 0.
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        label = F.one_hot(label, cfg.data.classes).float()
        loss = loss_fn(img, label)
        loss.backward()
        if cfg.optim.warmup > 0:
            for g in optim.param_groups:
                g['lr'] = cfg.optim.lr * np.minimum(epoch / cfg.optim.warmup, 1.0)
        if cfg.optim.grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optim.grad_clip)
        optim.step()
        optim.zero_grad()
        ema.update(model.parameters())
        loss_epoch += loss.item()
    if dist.get_rank() == 0:
        tb_writer.add_scalar("Loss/Train", loss_epoch/len(train_loader), epoch + 1)
    # eval
    if (epoch + 1) % cfg.train.eval_freq == 0:
        model.eval()
        with torch.no_grad():
            img, label = next(iter(test_loader))
            img, label = img.to(device), label.to(device)
            label = F.one_hot(label, cfg.data.classes).float()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            loss = loss_fn(img, label)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            ema.restore(model.parameters())
        if dist.get_rank() == 0:
                tb_writer.add_scalar("Loss/Eval", loss, epoch + 1)
    # sampling snapshot
    if (epoch + 1) % cfg.train.sampling_freq == 0 and dist.get_rank() == 0:
        model.eval()
        with torch.no_grad():
            noise = sde.prior_sampling([cfg.sampling.batch_size, *cfg.data.shape]).to(device)
            label = torch.arange(0, cfg.data.classes, device=device).repeat_interleave(cfg.sampling.batch_size // cfg.data.classes)
            label = F.one_hot(label.long(), cfg.data.classes).float()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            pred = sde.sampling(score_fn, noise=noise, label=label)
            ema.restore(model.parameters())
            fig = make_grid(pred, nrow=cfg.sampling.batch_size//cfg.data.classes)
            tb_writer.add_image(f"Sample", fig, epoch + 1)
    # save snapshot
    if (epoch + 1) % cfg.train.save_freq == 0:
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), optim=optim.state_dict(), epoch=epoch), \
                       f"logs/{cfg.meta.desc}/ckpt/{epoch+1}.pth")
