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

import argparse
import csv
import os
import time
from typing import Dict

import torch

from algos import train_gflownet
from envs import TSP

from exp.TSP.utils import (
    InstanceData,
    build_model,
    ensure_dir,
    load_config,
    load_dataset_lines,
    parse_instance,
    set_seed,
    summarize_tensor,
    flatten_config,
    write_json,
)


def forward_logprob_fn(model, trajectories: torch.Tensor) -> torch.Tensor:
    log_probs = model(trajectories)
    steps = trajectories.shape[1]
    return log_probs.unsqueeze(-1) / steps


def run_instance(
    instance_id: int,
    instance: InstanceData,
    master_config: Dict,
    flat_config: Dict,
    args: argparse.Namespace,
    base_results_dir: str,
) -> None:
    device = torch.device(master_config["model"]["device"])
    env = TSP(instance.coords.to(device=device))

    model = build_model(flat_config, env)
    optimizer = torch.optim.Adam(model.parameters(), lr=flat_config["lr"])

    instance_dir = os.path.join(base_results_dir, f"instance_{instance_id:03d}")
    ensure_dir(instance_dir)

    summary_path = os.path.join(instance_dir, "summary.json")

    gf_cfg = master_config["gflownet"]
    train_cfg = master_config["training"]
    warmup_steps = int(train_cfg.get("n_warmup", 0))
    anneal_steps = int(train_cfg.get("n_anneal", 0))
    total_steps = warmup_steps + anneal_steps if (warmup_steps + anneal_steps) > 0 else int(gf_cfg.get("n_steps", 0))

    start = time.perf_counter()
    result = train_gflownet(
        env,
        model,
        optimizer=optimizer,
        n_steps=total_steps if total_steps > 0 else int(gf_cfg["n_steps"]),
        batch_size=int(train_cfg["batch_size"]),
        temperature=float(gf_cfg["temperature"]),
        forward_logprob_fn=forward_logprob_fn,
        device=device,
        warmup_steps=warmup_steps,
        anneal_steps=anneal_steps,
    )
    training_time = time.perf_counter() - start

    sampling_start = time.perf_counter()
    with torch.no_grad():
        samples = model.sample(args.sample_count)
        energies = env.energy(samples.to(device=device))
    sampling_time = time.perf_counter() - sampling_start

    residuals = energies - instance.optimal_energy

    loss_hist = result["loss_history"].tolist()
    energy_mean_hist = result["energy_mean"].tolist()
    energy_min_hist = result["energy_min"].tolist()
    energy_max_hist = result["energy_max"].tolist()
    energy_var_hist = result["energy_var"].tolist()

    metrics_path = os.path.join(instance_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "mean_energy", "min_energy", "max_energy", "var_energy"])
        for idx, (loss, mean_e, min_e, max_e, var_e) in enumerate(
            zip(loss_hist, energy_mean_hist, energy_min_hist, energy_max_hist, energy_var_hist)
        ):
            writer.writerow([idx, loss, mean_e, min_e, max_e, var_e])

    summary = {
        "instance_id": instance_id,
        "dataset_path": os.path.abspath(args.dataset),
        "n_nodes": int(instance.coords.shape[0]),
        "seed": int(args.seed),
        "optimal_energy": instance.optimal_energy,
        "training_time_seconds": training_time,
        "sampling_time_seconds": sampling_time,
        "logZ": float(result["logZ"].item()),
        "loss_history": loss_hist,
        "energy_mean_history": energy_mean_hist,
        "energy_min_history": energy_min_hist,
        "energy_max_history": energy_max_hist,
        "energy_var_history": energy_var_hist,
        "energy_statistics": summarize_tensor(energies),
        "residual_statistics": summarize_tensor(residuals),
    }

    write_json(summary_path, summary)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    master_config = load_config(args.config)
    flat_config = flatten_config(master_config)
    flat_config["seed"] = args.seed
    expected_nodes = int(master_config["model"]["seq_size"])
    device = torch.device(master_config["model"]["device"])
    lines = load_dataset_lines(args.dataset)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results_gflownet")
    ensure_dir(results_dir)

    max_instances = min(args.max_instances, len(lines))
    for idx in range(max_instances):
        instance = parse_instance(lines[idx], expected_nodes, device)
        run_instance(idx, instance, master_config, flat_config, args, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP-200 GFlowNet experiment with extended logging.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--dataset", required=True, help="Path to the TSP dataset file.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--sample-count", type=int, default=2000, help="Number of samples for post-run evaluation.")
    parser.add_argument("--max-instances", type=int, default=10, help="Number of dataset instances to run.")
    main(parser.parse_args())
