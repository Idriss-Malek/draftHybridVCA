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
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))

from envs import SherringtonKirkpatrick
from models import DilatedRNN
from neighborModels import MLP as NeighborMLP
from supervisedLosses import *

import torch
import torch.optim as optim
import torch.nn.functional as F


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


def main(config, sk_data):
    record = {}
    device = torch.device(config["device"])
    J = read_matrix(sk_data).to(dtype=torch.float64, device=device)

    env = SherringtonKirkpatrick(J)
    if config["model_type"] == "DilatedRNN":
        model = DilatedRNN(config, env)
    if config["model_type"] == "NeighborMLP":
        model = NeighborMLP(config, env)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    if config["n_supervised"] is not None and config["n_supervised"] > 0:
        heuristic_samples_file = os.path.normpath(
            os.path.join(
                current_dir,
                "..",
                "heuristic_samples",
                config["heuristic"],
                "sk",
                f"{sk_data[-9:-4]}.pt",
            )
        )
        heuristic_samples_idx = torch.load(heuristic_samples_file, map_location=device)
        heuristic_samples = F.one_hot(
            ((heuristic_samples_idx +1)/2).to(dtype = torch.long), num_classes=2
        ).to(device=device, dtype=torch.float64)
        heuristic_energies = env.energy(heuristic_samples)
        supervisedTemp = torch.std(heuristic_energies, correction=0) / 2
        p_train = F.softmax(-heuristic_energies / supervisedTemp, dim = -1)

    T0 = config["T0"]

    for t in range(config["n_supervised"]):
        optimizer.zero_grad()
        heuristic_log_probs = model(heuristic_samples)
        if config.get('supervised_loss', None) is None:
            loss = torch.sum(p_train * (p_train.log() - heuristic_log_probs))
        else:
            if config['supervised_loss'] == 'forward_kl':
                loss = forward_kl(p_train, heuristic_log_probs)
            if config['supervised_loss'] == 'reverse_kl':
                loss = reverse_kl(p_train, heuristic_log_probs)
        loss.backward()
        optimizer.step()

    for _ in range(config["n_warmup"]):

        optimizer.zero_grad()
        solutions = model.sample(config["batch_size"])
        log_probs = model(solutions)
        energies = env.energy(solutions)
        FreeEnergy = energies + T0 * log_probs
        F_detached = FreeEnergy.detach()
        loss = torch.mean(log_probs * F_detached) - torch.mean(log_probs) * torch.mean(
            F_detached
        )
        loss.backward()
        optimizer.step()

    # with torch.no_grad():
    #     record[0] = float('inf')
    #     for _ in range(1000):
    #         solutions = model.sample(config["batch_size"])
    #         energies = env.energy(solutions)
    #         record[0] = min(torch.min(energies).item(), record[0])

    # Annealing step
    for t in range(config["n_anneal"]):
        T = T0 * (1 - t / config["n_anneal"])
        for _ in range(config["iter_per_temp"]):
            optimizer.zero_grad()
            solutions = model.sample(config["batch_size"])
            log_probs = model(solutions)
            energies = env.energy(solutions)
            FreeEnergy = energies + T * log_probs
            F_detached = FreeEnergy.detach()
            loss = torch.mean(log_probs * F_detached) - torch.mean(
                log_probs
            ) * torch.mean(F_detached)
            loss.backward()
            optimizer.step()
            
    record["mean"] = 0.0
    record["min"] = float("inf")
    with torch.no_grad():
        for st in range(1000):
            solutions = model.sample(config["batch_size"])
            energies = env.energy(solutions)
            record["min"] = min(record["min"], torch.min(energies).item())
            record["mean"] = (record["mean"] * st + torch.mean(energies).item()) / (st + 1)

    return record


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path. If omitted, a path is generated under ./results/.",
    )
    args = parser.parse_args()


    # Load config JSON -> dict
    with open(args.config, "r") as f:
        config = json.load(f)
        sk_instances_dir = _repo_root / "data" / "SK_Instances"
        sk_data = str(sk_instances_dir / f"100_SK_seed{config['seed']}")

    out_path = args.out or os.path.join(
        current_dir,
        "results", "sk",
        f"sk_{sk_data[-9:-4]}_{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_anneal", "min", "mean"])

    # Run
    for n_anneal in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        config["n_anneal"] = n_anneal
        result = main(config, sk_data)
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([n_anneal, result['min'], result['mean']])

        print(f"Saved results to {out_path}")
