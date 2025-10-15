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

import torch
import os
import csv

def VCA(model, env, config, optimizer, warmup=False, record=None):
    if record is not None:
        dirname = os.path.dirname(record)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        if not os.path.exists(record):
            with open(record, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "temperature", "minE", "maxE", "meanE", "VarE"])

    T0 = config['T0']
    steps = 1 if warmup else config["n_anneal"]

    for t in range(steps):
        T = T0 * (1 - t / config["n_anneal"])
        # T = T0 * (config["T_min"] / T0) ** (t / config["n_anneal"])
        inner_steps = config["n_warmup"] if warmup else config["iter_per_temp"]

        for _ in range(inner_steps):
            optimizer.zero_grad()
            solutions = model.sample(config["batch_size"])
            log_probs = model(solutions)              # log πθ(x)
            energies = env.energy(solutions)          # E(x)

            FreeEnergy = energies + T * log_probs     # F_T(x) = E + T log π
            F_detached = FreeEnergy.detach()

            loss = torch.mean(log_probs * F_detached) - torch.mean(log_probs) * torch.mean(F_detached)
            loss.backward()
            optimizer.step()

        if record is not None:
            with torch.no_grad():
                solutions = model.sample(config["batch_size"])
                energies = env.energy(solutions)

                minE = torch.min(energies).item()
                maxE = torch.max(energies).item()
                meanE = torch.mean(energies).item()
                varE = torch.var(energies, unbiased=False).item()

            with open(record, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([t + 1, T, minE, maxE, meanE, varE])
