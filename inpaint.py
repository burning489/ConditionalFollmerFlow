# %%
import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config.ode_mnist import cfg as cfg
from network import UNet
from ema import ExponentialMovingAverage
from sde import Follmer

device = torch.device("cuda:0")

# %%
ckpt_path = "logs/ode-mnist/ckpt/2000.pth"
model = UNet(**cfg.model).to(device)
ema = ExponentialMovingAverage(model.parameters())
ckpt = torch.load(ckpt_path, map_location=device)
ema.load_state_dict(ckpt["ema"])
ema.copy_to(model.parameters())
sde = Follmer()

def get_score_fn():
    def score_fn(x, t, label):
        # exaggerate temporal variable by T for numerical stability
        score = model(x, t*cfg.sampling.T, label)
        # scale output
        std = sde.marginal_prob(torch.empty_like(x, device=x.device), t)[1]
        return score / std[:, None, None, None]
    return score_fn
score_fn = get_score_fn()

# %%
mnist_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

mnist_dataloader = DataLoader(mnist_dataset, batch_size=1, shuffle=False)

tmp = list(range(10))
samples = torch.empty(10, 1, 28, 28, device=device)

for data, labels in mnist_dataloader:
    label = labels.item()
    if label in tmp:
        samples[label, ...] = data
        tmp.remove(int(label))
    if not tmp:
        break

# %%
samples_grid = make_grid(samples, nrow=1)
# plt.imshow(samples_grid.cpu().permute(1, 2, 0))
# plt.axis("off")

# %%
mask_ver = 3
if mask_ver == 1:
    mask = torch.zeros_like(samples, device=device)
    mask[:, :, 14:, :14] = 1.
elif mask_ver == 2:
    mask = torch.ones_like(samples, device=device)
    mask[:, :, :, 14:] = 0.
elif mask_ver == 3:
    mask = torch.ones_like(samples, device=device)
    mask[:, :, 14:, 14:] = 0.
masked_samples_grid = make_grid(samples*mask, nrow=1)
# plt.imshow(masked_samples_grid.cpu().permute(1, 2, 0))
# plt.axis("off")

# %%
examples_per_digit = 10
reference = samples.repeat([examples_per_digit, 1, 1, 1]).reshape(examples_per_digit, 10, *samples.shape[1:]).transpose(1, 0).reshape(-1, *samples.shape[1:])
mask = mask.repeat([examples_per_digit, 1, 1, 1]).reshape(examples_per_digit, 10, *samples.shape[1:]).transpose(1, 0).reshape(-1, *samples.shape[1:])

model.eval()
with torch.no_grad():
    noise = sde.prior_sampling([examples_per_digit*10, *cfg.data.shape]).to(device)
    pred = sde.sampling(score_fn, noise=noise, reference=reference, mask=mask)
    pred_grid = make_grid(pred, nrow=examples_per_digit)
    # plt.imshow(pred_grid.clamp(0.0, 1.0).cpu().permute(1, 2, 0))
    # plt.axis("off")

# %%
grid = torch.concat([samples_grid, masked_samples_grid, pred_grid], dim=2)
plt.imshow(grid.clamp(0.0, 1.0).cpu().permute(1, 2, 0))
plt.axis("off")
plt.savefig(f"asset/mnist-inpaint-{mask_ver}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.)
