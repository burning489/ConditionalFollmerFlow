# This code is adapted from https://github.com/rtqichen/ffjord.
# It can generate 2D toy datasets.
import os
import argparse
import pickle

import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle

from misc import add_dict_to_argparser


def generate_toy_data(target_name, nsample=2000, rng=None) -> np.ndarray:
    if rng is None or isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        raise TypeError(f"unsupported type of rng{type(rng)}, supported: None or integer")

    if target_name == "gaussian":
        target = rng.normal(0, 1, size=(nsample, 2))
        target = target
        return target

    if target_name == "swissroll":
        target = sklearn.datasets.make_swiss_roll(n_samples=nsample, noise=1.0, random_state=rng)[0]
        target = target[:, [0, 2]]
        target /= 5
        return target

    elif target_name == "circles":
        target = sklearn.datasets.make_circles(n_samples=nsample, factor=0.5, noise=0.08, random_state=rng)[0]
        target = target
        target *= 3
        return target # pyright: ignore

    elif target_name == "small_circle":
        theta = rng.uniform(0, 1, size=nsample)
        radius = rng.normal(0, 0.1, size=nsample) + 1.2
        target = np.array([radius * np.cos(2 * np.pi * theta), radius * np.sin(2 * np.pi * theta)]).T
        target = target
        return target

    elif target_name == "large_circle":
        theta = rng.uniform(0, 1, size=nsample)
        radius = rng.normal(0, 0.1, size=nsample) + 3
        target = np.array([radius * np.cos(2 * np.pi * theta), radius * np.sin(2 * np.pi * theta)]).T
        target = target
        return target

    elif target_name == "large_4gaussians":
        scale = 3
        theta = rng.randint(0, 4, size=nsample) / 4
        target = np.array([scale * np.cos(2 * np.pi * theta - np.pi / 4), scale * np.sin(2 * np.pi * theta - np.pi / 4),]).T + rng.randn(nsample, 2) * 0.3
        target = np.array(target, dtype="float32")
        return target

    elif target_name == "small_4gaussians":
        scale = 1.5
        theta = rng.randint(0, 4, size=nsample) / 4
        target = np.array([scale * np.cos(2 * np.pi * theta - np.pi / 4), scale * np.sin(2 * np.pi * theta - np.pi / 4),]).T + rng.randn(nsample, 2) * 0.3
        target = np.array(target, dtype="float32")
        return target

    elif target_name == "rings":
        n_samples4 = n_samples3 = n_samples2 = nsample // 4
        n_samples1 = nsample - n_samples4 - n_samples3 - n_samples2
        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)
        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25
        X = np.vstack([np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]), np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])]).T * 3.0
        X = shuffle(X, random_state=rng)
        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape) # pyright: ignore
        return X

    elif target_name == "moons":
        target = sklearn.datasets.make_moons(n_samples=nsample, noise=0.1, random_state=rng)[0]
        target = target
        target = target * 2 + np.array([-1, -0.2])
        return target

    elif target_name == "8gaussians":
        scale = 3
        theta = rng.randint(0, 8, size=nsample) / 8
        target = np.array([scale * np.cos(2 * np.pi * theta), scale * np.sin(2 * np.pi * theta)]).T + rng.randn(nsample, 2) * 0.3
        target = np.array(target, dtype="float32")
        return target

    elif target_name == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = nsample // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        features = rng.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)
        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        ret = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        return ret

    elif target_name == "2spirals":
        n = np.sqrt(rng.rand(nsample // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.rand(nsample // 2, 1) * 0.5
        d1y = np.sin(n) * n + rng.rand(nsample // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += rng.randn(*x.shape) * 0.1
        return x

    elif target_name == "spiral1":
        n = np.sqrt(rng.rand(nsample, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.rand(nsample, 1) * 0.5
        d1y = np.sin(n) * n + rng.rand(nsample, 1) * 0.5
        x = np.hstack((d1x, d1y)) / 3
        x += rng.randn(*x.shape) * 0.1
        return x

    elif target_name == "spiral2":
        n = np.sqrt(rng.rand(nsample, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.rand(nsample, 1) * 0.5
        d1y = np.sin(n) * n + rng.rand(nsample, 1) * 0.5
        x = np.hstack((-d1x, -d1y)) / 3
        x += rng.randn(*x.shape) * 0.1
        return x

    elif target_name == "checkerboard":
        x1 = rng.rand(nsample) * 4 - 2
        x2_ = rng.rand(nsample) - rng.randint(0, 2, nsample) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        ret = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        return ret

    elif target_name == "5squares":
        idx = rng.randint(0, 5, nsample)
        idx_zo = 1 - idx // 4
        x1 = (rng.rand(nsample) - 1 / 2) + idx_zo * (rng.randint(0, 2, nsample) * 4 - 2)
        x2 = (rng.rand(nsample) - 1 / 2) + idx_zo * (rng.randint(0, 2, nsample) * 4 - 2)
        ret = np.concatenate([x1[:, None], x2[:, None]], 1)
        return ret

    elif target_name == "4squares":
        idx = rng.randint(0, 2, nsample)
        x1 = (rng.rand(nsample) - 1 / 2) + idx * (rng.randint(0, 2, nsample) * 4 - 2)
        x2 = (rng.rand(nsample) - 1 / 2) + (1 - idx) * (rng.randint(0, 2, nsample) * 4 - 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1)

    elif target_name == "line":
        x = rng.rand(nsample) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif target_name == "cos":
        x = rng.rand(nsample) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        raise ValueError(f"Wrong dataset target_name: {target_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    defaults = dict(data='moons', n_train=50000, n_test=5000, n_valid=5000, seed=42, output='dataset/moons.pkl')
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()
    data = generate_toy_data(args.data, args.n_train+args.n_test+args.n_valid, args.seed).astype(np.float32)
    x, y = data[:, 0:1], data[:, 1:2]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(dict(
            x_train=x[:args.n_train],
            y_train=y[:args.n_train],
            x_test=x[args.n_train:args.n_train+args.n_test],
            y_test=y[args.n_train:args.n_train+args.n_test],
            x_valid=x[-args.n_valid:],
            y_valid=y[-args.n_valid:],
        ), f)
