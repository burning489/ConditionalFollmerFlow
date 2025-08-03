import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import argparse
import torch

from misc import add_dict_to_argparser


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(rundir='', num_classes=10, n_per_class=10)
    add_dict_to_argparser(parser, defaults)
    return parser


def main(args):
	x = np.load(os.path.join(args.rundir, 'samples.npy'))
	N_per_class = x.shape[0] // args.num_classes
	idx = [class_idx*N_per_class+i for class_idx in range(args.num_classes) for i in range(args.n_per_class)]
	x = x[idx]
	grid = make_grid(torch.tensor(x), nrow=args.n_per_class)
	plt.imshow(grid.permute(1, 2, 0))
	plt.axis("off")
	plt.savefig(os.path.join(args.rundir, f'samples.png'), dpi=300, bbox_inches="tight", pad_inches=0.)


if __name__ == '__main__':
    args = create_parser().parse_args()
    main(args)
