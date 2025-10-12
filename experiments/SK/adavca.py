import sys
import os
import argparse
import json
import csv
from datetime import datetime
import copy
import random
import numpy as np

sys.path.append("/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA")
current_dir = os.path.dirname(os.path.abspath(__file__))

from models import DilatedRNN, VanillaRNN
from supervisedLosses import *

import torch
import torch.optim as optim
import torch.nn.functional as F


from envs import SherringtonKirkpatrick


from algos import VCA, adaVCA



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

def read_matrix(path: str) -> torch.Tensor:
    triples = []
    N = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Expected 'i j value' per line, got: {line}")
            i, j = int(parts[0]), int(parts[1])
            v = float(parts[2])
            triples.append((i, j, v))
            if i > N: N = i
            if j > N: N = j
    J = torch.zeros((N, N), dtype=torch.float64)
    for i, j, v in triples:
        i0, j0 = i - 1, j - 1  
        if i0 == j0:
            continue
        J[i0, j0] = v
        J[j0, i0] = v  

    J.fill_diagonal_(0.0)
    return J

def main(config, seed: int):
    set_seed(seed)
    device = torch.device(config["device"])
    sk_data = f"/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/SK_Instances/100_SK_seed{seed}.txt"
    J = read_matrix(sk_data).to(dtype=torch.float64, device=device)

    env = SherringtonKirkpatrick(J)

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
    record_path = os.path.join(results_dir, f"adavca{seed}.csv")

    # Annealing — write results to current_dir/results/vca{seed}.csv
    adaVCA(model, env, config, optimizer, record=record_path)

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
