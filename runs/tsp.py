import sys
import os
import argparse
import json
import csv
import os
from datetime import datetime

sys.path.append("/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA")
current_dir = os.path.dirname(os.path.abspath(__file__))

from envs import TSP
from models import DilatedRNN

import torch
import torch.optim as optim
import torch.nn.functional as F

tsp_data = {
    64: "/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/TSP Instances/coordinates_N64.txt",
    128: "/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/TSP Instances/coordinates_N128.txt",
    256: "/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/TSP Instances/coordinates_N256.txt",
}


def main(config):
    record = {}
    coords = []
    device = torch.device(config["device"])
    with open(tsp_data[config["seq_size"]], "r") as f:
        lines = f.readlines()
        n = int(lines[0].strip())  # first line = number of coordinates
        for line in lines[1:]:
            x, y = map(float, line.strip().split())
            coords.append([x, y])

    coordinates = torch.tensor(coords, dtype=torch.float64, device=device)
    env = TSP(coordinates)
    if config["model_type"] == "DilatedRNN":
        model = DilatedRNN(config, env)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    if config["n_supervised"] is not None and config["n_supervised"] > 0:
        heuristic_samples_file = os.path.normpath(
            os.path.join(
                current_dir,
                "..",
                "heuristic_samples",
                config["heuristic"],
                "tsp",
                f"tsp{config['seq_size']}anneal1024.pt",
            )
        )
        heuristic_samples_idx = torch.load(heuristic_samples_file, map_location=device)
        heuristic_samples = F.one_hot(
            heuristic_samples_idx, num_classes=heuristic_samples_idx.size(1)
        ).to(device=device, dtype=torch.float64)
        heuristic_energies = env.energy(heuristic_samples)
        supervisedTemp = torch.std(heuristic_energies, correction=0) / 2
        p_train = F.softmax(-heuristic_energies / supervisedTemp, dim = -1)

    T0 = config["T0"]

    for t in range(config["n_supervised"]):
        heuristic_log_probs = model(heuristic_samples)
        loss = torch.sum(p_train * (p_train.log() - heuristic_log_probs))

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

    parser = argparse.ArgumentParser(
        description="Train NeighborVCA model on TSP and save results."
    )
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

    out_path = args.out or os.path.join(
        current_dir,
        "results", "tsp",
        f"tsp{config['seq_size']}_{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_anneal", "min", "mean"])

    # Run
    for n_anneal in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        config["n_anneal"] = n_anneal
        result = main(config)
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([n_anneal, result['min'], result['mean']])

        print(f"Saved results to {out_path}")
