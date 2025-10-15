import sys
from pathlib import Path

_repo_root = Path(__file__).resolve()
for parent in _repo_root.parents:
    if (parent / ".gitignore").exists():
        _repo_root = parent
        break
else:
    _repo_root = _repo_root.parent

_repo_root_str = str(_repo_root)
if _repo_root_str not in sys.path:
    sys.path.append(_repo_root_str)

import sys
import os
import argparse
import json
import csv
from datetime import datetime
import copy
import random
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

from envs import Portfolio
from models import DilatedRNN, VanillaRNN
from supervisedLosses import *

import torch
import torch.optim as optim
import torch.nn.functional as F

from algos import VCA

def set_seed(seed: int):
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For more reproducibility (may have perf cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(config, seed: int):
    set_seed(seed)

    env, Estar = Portfolio.from_random(600, seed = seed)

    if config["model_type"] == "DilatedRNN":
        model = DilatedRNN(config, env)
    elif config["model_type"] == "VanillaRNN":
        model = VanillaRNN(config, env)
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Warmup
    VCA(model, env, config, optimizer, warmup=True, record=None)

    # Ensure results directory exists and set record path
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    record_path = os.path.join(results_dir, f"vca{seed}.csv")

    # Annealing — write results to current_dir/results/vca3.csv
    VCA(model, env, config, optimizer, warmup=False, record=record_path)
    df = pd.read_csv(record_path)
    cols_to_subtract = ["minE", "maxE", "meanE"]
    df[cols_to_subtract] = df[cols_to_subtract] - Estar
    df.to_csv(record_path, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path. If omitted, a path is generated under ./results/.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (int).")
    args = parser.parse_args()

    # Load config JSON -> dict
    with open(args.config, "r") as f:
        config = json.load(f)

    # Run with provided seed
    main(config, args.seed)
