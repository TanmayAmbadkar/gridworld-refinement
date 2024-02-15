import argparse
import random
import numpy
import torch
from four_grid import run_4grid
from three_grid import run_3grid
import os

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-e", "--exp-id", type=int, default=0)
parser.add_argument("-g", "--grid-size", type=int, default=3)
parser.add_argument("-n", "--n-episodes", type=int, default=5000)
parser.add_argument("-m", "--min-reach", type=float, default=0.9)

if __name__ == "__main__":
    args = parser.parse_args()
    print("seed:", args.seed)
    print("exp_id:", args.exp_id)
    print("grid_size:", args.grid_size)
    
    path = f"results/{args.grid_size}_grid-exp_{args.exp_id}-n_ep_{args.n_episodes}-seed_{args.seed}"
    if not os.path.exists(path):
        os.mkdir(path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    if args.grid_size == 3:
        run_3grid(args.min_reach, args.n_episodes)
    else:
        run_4grid(args.min_reach, args.n_episodes)