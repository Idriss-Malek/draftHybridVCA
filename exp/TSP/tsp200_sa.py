import argparse
import csv
import os
import time
from typing import Dict

import torch
import torch.nn.functional as F

from algos import simulated_annealing
from envs import TSP

from exp.TSP.utils import (
    InstanceData,
    ensure_dir,
    load_config,
    load_dataset_lines,
    parse_instance,
    set_seed,
    summarize_tensor,
    write_json,
)


def random_permutation_matrices(batch_size: int, n: int, device: torch.device) -> torch.Tensor:
    perms = torch.stack([torch.randperm(n, device=device) for _ in range(batch_size)], dim=0)
    return F.one_hot(perms, num_classes=n).to(dtype=torch.float64)


def run_instance(
    instance_id: int,
    instance: InstanceData,
    master_config: Dict,
    args: argparse.Namespace,
    base_results_dir: str,
) -> None:
    device = torch.device(master_config["model"].get("device", "cpu"))
    env = TSP(instance.coords.to(device=device))

    sa_cfg = master_config["simulated_annealing"]
    train_cfg = master_config["training"]
    warmup_override = int(train_cfg.get("n_warmup", sa_cfg.get("warmup_steps", 0)))
    anneal_override = int(train_cfg.get("n_anneal", sa_cfg.get("n_temps", 0)))
    iter_per_temp = int(train_cfg.get("iter_per_temp", sa_cfg.get("steps_per_temp", 1)))

    batch_size = int(sa_cfg.get("batch_size", args.sample_count))
    batch_size = max(batch_size, args.sample_count)

    initial_state = random_permutation_matrices(batch_size, instance.coords.shape[0], device)

    sa_kwargs = {
        "schedule": sa_cfg.get("schedule", "geometric"),
        "n_temps": anneal_override if anneal_override > 0 else int(sa_cfg.get("n_temps", 200)),
        "steps_per_temp": iter_per_temp,
        "T0": float(sa_cfg.get("T0", 2.0)),
        "Tend": float(sa_cfg.get("Tend", 1e-3)),
        "warmup_steps": warmup_override,
        "seed": sa_cfg.get("seed"),
        "canonicalize": bool(sa_cfg.get("canonicalize", False)),
        "return_traces": True,
    }

    instance_dir = os.path.join(base_results_dir, f"instance_{instance_id:03d}")
    ensure_dir(instance_dir)
    summary_path = os.path.join(instance_dir, "summary.json")

    start = time.perf_counter()
    result = simulated_annealing(
        env,
        init_solution=initial_state,
        neighbor_arg_sampler=None,
        **sa_kwargs,
    )
    run_time = time.perf_counter() - start

    samples = result["best_state"]
    energies = env.energy(samples.to(device=device))
    residuals = energies - instance.optimal_energy

    trace = result.get("trace_best")
    metrics_path = os.path.join(instance_dir, "metrics.csv")
    if trace is not None:
        trace_list = trace.detach().cpu().tolist()
        warmup_steps = int(sa_cfg.get("warmup_steps", 0))
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "best_energy", "stage", "mean_energy", "min_energy", "max_energy", "var_energy"])
            for idx, value in enumerate(trace_list):
                stage = "warmup" if idx < warmup_steps else "anneal"
                writer.writerow([idx, value, stage, value, value, value, 0.0])

            sample_stats = summarize_tensor(energies)
            writer.writerow([
                "final",
                sample_stats["min"],
                "final",
                sample_stats["mean"],
                sample_stats["min"],
                sample_stats["max"],
                sample_stats["std"] ** 2,
            ])

    summary = {
        "instance_id": instance_id,
        "dataset_path": os.path.abspath(args.dataset),
        "n_nodes": int(instance.coords.shape[0]),
        "seed": int(args.seed),
        "optimal_energy": instance.optimal_energy,
        "runtime_seconds": run_time,
        "energy_statistics": summarize_tensor(energies),
        "residual_statistics": summarize_tensor(residuals),
        "trace_length": int(result.get("trace_best", torch.empty(0)).numel()),
    }

    write_json(summary_path, summary)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    master_config = load_config(args.config)
    expected_nodes = int(master_config["model"].get("seq_size", 200))
    device = torch.device(master_config["model"].get("device", "cpu"))
    lines = load_dataset_lines(args.dataset)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results_sa")
    ensure_dir(results_dir)

    max_instances = min(args.max_instances, len(lines))
    for idx in range(max_instances):
        instance = parse_instance(lines[idx], expected_nodes, device)
        run_instance(idx, instance, master_config, args, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP-200 Simulated Annealing experiment with extended logging.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--dataset", required=True, help="Path to the TSP dataset file.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--sample-count", type=int, default=2000, help="Number of samples for post-run evaluation.")
    parser.add_argument("--max-instances", type=int, default=10, help="Number of dataset instances to run.")
    main(parser.parse_args())
