import argparse
import os
import pickle

import numpy as np
import torch
from misc import add_dict_to_argparser


def m1(size: int):
    y = np.random.randn(size, 5)
    mean = y[:, :1]**2 + np.exp(y[:, 1:2] + y[:, 2:3]/4) + np.cos(y[:, 3:4] + y[:, 4:])
    x = mean + np.random.randn(*mean.shape)
    std=np.ones_like(mean)
    return dict(y=y, x=x, mean=mean, std=std)

def m2(size: int):
    y = np.random.randn(size, 5)
    mean = y[:, :1]**2 + np.exp(y[:, 1:2] + y[:, 2:3]/4) + y[:, 3:4] - y[:, 4:] 
    std = (1 + y[:, 1:2]**2 + y[:, 4:]**2) / 2 
    x = mean + std * np.random.randn(*mean.shape)
    return dict(y=y, x=x, mean=mean, std=std)

def m3(size: int):
    y = np.random.randn(size, 50)
    std = 0.1 * np.sum(y[:, 0::5] * y[:, 1::5] * np.cos(2 * y[:, 2::5] * y[:, 3::5] + y[:, 4::5]), axis=1,
                     keepdims=True) + 1
    mean = 0.1 * np.sum((y[:, 0::5] + y[:, 1::5] - 1)**2 + y[:, 2::5] * np.sin(y[:, 3::5] + 3 * y[:, 4::5]), axis=1, keepdims=True)
    x = mean + std * np.random.randn(size, 1)
    return dict(y=y, x=x, mean=mean, std=std)

def get_target_fn(data_name):
    return dict(m1=m1, m2=m2, m3=m3)[data_name]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    defaults = dict(data='m1', n_train=50000, n_test=5000, n_valid=5000, seed=42, output='dataset/m1.pkl')
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()
    np.random.seed(args.seed)
    data = get_target_fn(args.data)(args.n_train+args.n_test+args.n_valid)
    x = torch.from_numpy(data['x']).float()
    y = torch.from_numpy(data['y']).float()
    mean = torch.from_numpy(data['mean']).float() # sample-wise, not feature-wise
    std = torch.from_numpy(data['std']).float() # sample-wise, not feature-wise
    x_train, y_train = x[:args.n_train], y[:args.n_train]
    x_test, y_test = x[args.n_train:args.n_train+args.n_test], y[args.n_train:args.n_train+args.n_test]
    x_valid, y_valid = x[-args.n_valid:], y[-args.n_valid:]
    mean_train, std_train = mean[:args.n_train], std[:args.n_train]
    mean_test, std_test = mean[args.n_train:args.n_train+args.n_test], std[args.n_train:args.n_train+args.n_test]
    mean_valid, std_valid = mean[-args.n_valid:], std[-args.n_valid:]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(dict(
            x_train=x_train, y_train=y_train, mean_train=mean_train, std_train=std_train,
            x_test=x_test, y_test=y_test, mean_test=mean_test, std_test=std_test,
            x_valid=x_valid, y_valid=y_valid, mean_valid=mean_valid, std_valid=std_valid,
        ), f)