import argparse
import pickle

import torch
import numpy as np
import torch

from diffusion import Follmer

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, help="Path to network ckpt")
parser.add_argument("--data", type=str, help="Dataset name")
parser.add_argument("--ode_solver", type=str, help="ODE solver", default="euler")
parser.add_argument("--eps", type=float, help="Start time", default=1e-3)
parser.add_argument("--T", type=float, help="End time", default=0.999)
parser.add_argument("--N", type=int, help="Number of steps", default=100)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("cuda is not available")
device = torch.device("cuda:0")

with open(f'dataset/{args.data}.pkl', 'rb') as f:
    data = pickle.load(f)
x_test, y_test = data['x_test'], data['y_test']
x_test, y_test = torch.from_numpy(x_test).to(device), torch.from_numpy(y_test).to(device)

model = torch.load(args.ckpt, map_location=device)
model.eval()
sde = Follmer(args)

noise = sde.prior_sampling(shape=[y_test.shape[0], 1], device=device)
x_pred = sde.solve(model, noise, y_test)
x_pred = x_pred.cpu().numpy()
y_pred = y_test.cpu().numpy()
np.savez(f"logs/{args.data}-{args.T}.npz", x=x_pred, y=y_pred)
