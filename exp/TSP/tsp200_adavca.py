import argparse
import os
import time
from typing import Dict

import torch

from algos import VCA, adaVCA
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

    record_path = os.path.join(instance_dir, "adavca.csv")
    summary_path = os.path.join(instance_dir, "summary.json")

    start = time.perf_counter()
    VCA(model, env, flat_config, optimizer, warmup=True, record=None)
    adaVCA(model, env, flat_config, optimizer, record=record_path)
    training_time = time.perf_counter() - start

    sampling_start = time.perf_counter()
    with torch.no_grad():
        samples = model.sample(args.sample_count)
        energies = env.energy(samples.to(device=device))
    sampling_time = time.perf_counter() - sampling_start

    residuals = energies - instance.optimal_energy

    summary = {
        "instance_id": instance_id,
        "dataset_path": os.path.abspath(args.dataset),
        "n_nodes": int(instance.coords.shape[0]),
        "seed": int(args.seed),
        "optimal_energy": instance.optimal_energy,
        "training_time_seconds": training_time,
        "sampling_time_seconds": sampling_time,
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
    results_dir = os.path.join(current_dir, "results_adavca")
    ensure_dir(results_dir)

    max_instances = min(args.max_instances, len(lines))
    for idx in range(max_instances):
        instance = parse_instance(lines[idx], expected_nodes, device)
        run_instance(idx, instance, master_config, flat_config, args, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP-200 adaVCA experiment with extended logging.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--dataset", required=True, help="Path to the TSP dataset file.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--sample-count", type=int, default=2000, help="Number of samples for post-run evaluation.")
    parser.add_argument("--max-instances", type=int, default=10, help="Number of dataset instances to run.")
    main(parser.parse_args())
