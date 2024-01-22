# %%
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from config.ode_mnist_cond import cfg as cfg
from network import UNet
from ema import ExponentialMovingAverage
from sde import Follmer

device = torch.device("cuda:0")

# %% Plot MNIST Train Set Examples
mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor())

mnist_dataloader = DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=False)
images, labels = next(iter(mnist_dataloader))

classes = 10
examples_per_digit = 10

examples = torch.empty([classes, examples_per_digit, *cfg.data.shape], device=device)
for digit in range(10):
    digit_indices = (labels == digit).nonzero()[:, 0]
    digit_indices = digit_indices[:examples_per_digit]  
    examples[digit, ...] = images[digit_indices]
examples = examples.reshape(-1, *cfg.data.shape)
examples_grid = make_grid(examples, nrow=examples_per_digit)
plt.imshow(examples_grid.cpu().permute(1, 2, 0))
plt.axis("off")
plt.savefig(f"asset/mnist-example.pdf", dpi=300, bbox_inches="tight", pad_inches=0.)

# %% Plot Class-Conditional Generation Samples
ckpt_path = "logs/ode-mnist-cond/ckpt/2000.pth"
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

label = [F.one_hot(i*torch.ones((examples_per_digit, ), device=device).long(), classes).float() for i in range(classes)]
label = torch.cat(label, dim=0)

model.eval()
with torch.no_grad():
    noise = sde.prior_sampling([examples_per_digit*classes, *cfg.data.shape]).to(device)
    pred = sde.sampling(score_fn, noise=noise, label=label)
    pred_grid = make_grid(pred, nrow=examples_per_digit)
    plt.imshow(pred_grid.clamp(0.0, 1.0).cpu().permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(f"asset/mnist-class-cond.pdf", dpi=300, bbox_inches="tight", pad_inches=0.)
