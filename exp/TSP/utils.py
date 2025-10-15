from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from envs import TSP
from models import DilatedRNN, VanillaRNN


@dataclass
class InstanceData:
    coords: torch.Tensor
    optimal_tour: torch.Tensor
    optimal_energy: float


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_instance(line: str, n_nodes: int, device: torch.device) -> InstanceData:
    tokens = line.strip().split()
    expected_coord_tokens = 2 * n_nodes
    if len(tokens) < expected_coord_tokens + 1:
        raise ValueError("Malformed line: insufficient tokens.")

    coords = torch.tensor(
        list(map(float, tokens[:expected_coord_tokens])),
        dtype=torch.float64,
        device=device,
    ).view(n_nodes, 2)

    marker = tokens[expected_coord_tokens]
    if marker.lower() != "output":
        raise ValueError(f"Expected 'output' marker, found '{marker}'.")

    tour_tokens = list(map(int, tokens[expected_coord_tokens + 1 :]))
    if not tour_tokens:
        raise ValueError("No tour data provided.")
    tour = torch.tensor(tour_tokens, dtype=torch.long, device=device) - 1

    if tour.numel() == n_nodes + 1 and tour[0] == tour[-1]:
        tour = tour[:-1]

    if tour.numel() != n_nodes:
        raise ValueError(f"Tour length {tour.numel()} does not match n_nodes {n_nodes}.")

    if set(tour.tolist()) != set(range(n_nodes)):
        raise ValueError("Tour does not cover every node exactly once.")

    env = TSP(coords)
    optimal_energy = compute_optimal_energy(env, tour)

    return InstanceData(coords=coords, optimal_tour=tour, optimal_energy=optimal_energy)


def compute_optimal_energy(env: TSP, tour: torch.Tensor) -> float:
    n = tour.numel()
    perm = torch.zeros(n, n, dtype=torch.float64, device=env.coordinates.device)
    perm[torch.arange(n, device=env.coordinates.device), tour] = 1.0
    energy = env.energy(perm.unsqueeze(0))
    return float(energy.item())


def summarize_tensor(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().cpu()
    summary = {
        "count": int(values.numel()),
        "min": float(torch.min(values).item()),
        "max": float(torch.max(values).item()),
        "mean": float(torch.mean(values).item()),
        "std": float(torch.std(values, unbiased=False).item()),
        "median": float(torch.median(values).item()),
    }
    quantiles = torch.quantile(
        values,
        torch.tensor([0.25, 0.75], dtype=values.dtype)
    )
    summary["quantile_25"] = float(quantiles[0].item())
    summary["quantile_75"] = float(quantiles[1].item())
    return summary


def build_model(config: Dict, env: TSP):
    model_type = config.get("model_type", "DilatedRNN")
    if model_type == "DilatedRNN":
        return DilatedRNN(config, env)
    if model_type == "VanillaRNN":
        return VanillaRNN(config, env)
    raise ValueError(f"Unknown model_type: {model_type}")


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def load_dataset_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def flatten_config(config: Dict) -> Dict:
    order = ["model", "training", "adavca", "gflownet", "adagfn", "simulated_annealing"]
    flat: Dict = {}

    def merge(section: Dict) -> None:
        for key, value in section.items():
            if key not in flat:
                flat[key] = value

    for key in order:
        section = config.get(key)
        if isinstance(section, dict):
            merge(section)
    for key, value in config.items():
        if key in order:
            continue
        if isinstance(value, dict):
            merge(value)
        else:
            flat.setdefault(key, value)
    return flat


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
