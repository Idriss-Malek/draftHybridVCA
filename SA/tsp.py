from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F

import sys
sys.path.append('/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA')

from envs import TSP

@torch.no_grad()
def simulated_annealing_tsp(
    env,                          
    init_solution: Optional[torch.Tensor] = None,  
    batch_size: int = 128,
    warmup_steps: int = 0,
    n_temps: int = 200,
    steps_per_temp: int = 250,
    T0: float = 10.0,
    Tend: float = 1e-3,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    return_traces: bool = False,
) -> Dict[str, Any]:
    """
    Simulated annealing for TSP using 2-opt neighbors on permutation matrices.
    Warm up at T0, then anneal over a geometric temperature ladder down to Tend.
    Returns best per-chain solutions, the global best, and (optionally) a trace of the running global best.
    """
    assert n_temps >= 1 and steps_per_temp >= 0 and warmup_steps >= 0
    dtype = env.coordinates.dtype
    if device is None:
        device = env.coordinates.device

    # RNG (deterministic if seed is provided)
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)
        rand = lambda *shape: torch.rand(*shape, device=device, generator=g)
        randint = lambda low, high, shape: torch.randint(low, high, shape, device=device, generator=g)
    else:
        rand = lambda *shape: torch.rand(*shape, device=device)
        randint = lambda low, high, shape: torch.randint(low, high, shape, device=device)

    n = env.N
    B = init_solution.shape[0] if init_solution is not None else batch_size

    # Helpers
    def random_perm_matrices(B, n):
        # Batched random permutations via argsort of uniform noise
        noise = rand(B, n)
        perm_idx = noise.argsort(dim=1)
        return F.one_hot(perm_idx, num_classes=n).to(dtype=dtype)

    def idx_from_perm(P):
        return P.argmax(dim=2)

    # Init population
    if init_solution is None:
        P = random_perm_matrices(B, n)
    else:
        assert init_solution.shape == (B, n, n)
        P = init_solution.to(device=device, dtype=dtype)

    E = env.energy(P)
    P_best = P.clone()
    E_best = E.clone()

    # Track global best across the batch
    best_idx = torch.argmin(E)
    global_best = {
        "perm_matrix": P[best_idx].clone(),
        "tour_index": idx_from_perm(P[best_idx : best_idx + 1])[0].clone(),
        "energy": E[best_idx].item(),
    }

    total_steps = warmup_steps + n_temps * steps_per_temp
    trace_best = torch.empty(total_steps, device=device, dtype=dtype) if return_traces else None
    step_counter = 0

    # One Metropolis–Hastings proposal per chain at temperature T
    def mh_step_at_temperature(T: torch.Tensor):
        nonlocal P, E, P_best, E_best, global_best

        # Sample (i, j) with j != i
        i = randint(0, n, (B,))
        u = randint(0, n - 1, (B,))
        j = u + (u >= i)

        # Propose 2-opt neighbor and compute energy difference
        P_prop = env.neighbor(P, i, j)
        E_prop = env.energy(P_prop)
        dE = E_prop - E

        # Metropolis accept using log-test; clamp T to avoid divide-by-zero
        T_safe = torch.clamp(T, min=torch.finfo(dtype).tiny)
        accept = (dE <= 0) | (torch.log(rand(B)) < (-dE / T_safe))

        if accept.any():
            mask = accept.view(B, 1, 1)
            P = torch.where(mask, P_prop, P)
            E = torch.where(accept, E_prop, E)

        # Update per-chain and global bests
        better = E < E_best
        if better.any():
            P_best = torch.where(better.view(B, 1, 1), P, P_best)
            E_best = torch.where(better, E, E_best)

            cand_idx = torch.argmin(E)
            cand_energy = E[cand_idx].item()
            if cand_energy < global_best["energy"]:
                global_best["perm_matrix"] = P[cand_idx].clone()
                global_best["tour_index"] = idx_from_perm(P[cand_idx : cand_idx + 1])[0].clone()
                global_best["energy"] = cand_energy

    # Warm-up at fixed T0
    T0_tensor = torch.tensor(T0, device=device, dtype=dtype)
    for _ in range(warmup_steps):
        for _ in range(n):
            mh_step_at_temperature(T0_tensor)
        if return_traces:
            trace_best[step_counter] = torch.tensor(global_best["energy"], device=device, dtype=dtype)
        step_counter += 1

    # Linear: T_k = T0 + k*(Tend - T0)/(n_temps-1)
    if n_temps == 1:
        temps = torch.tensor([T0], device=device, dtype=dtype)
    else:
        temps = torch.linspace(T0, Tend, steps=n_temps, device=device, dtype=dtype)
        temps = torch.clamp(temps, min=torch.finfo(dtype).tiny)  


    # Equilibrate for 'steps_per_temp' steps at each temperature
    for Tk in temps:
        for _ in range(steps_per_temp):
            for __ in range(n):
                mh_step_at_temperature(Tk)
            if return_traces:
                trace_best[step_counter] = torch.tensor(global_best["energy"], device=device, dtype=dtype)
            step_counter += 1

    idx_best = idx_from_perm(P_best)

    out = {
        "best_perm_matrix": P_best,
        "best_tour_index": idx_best,
        "best_energy": E_best,
        "global_best": global_best,
    }
    if return_traces:
        out["trace_best"] = trace_best.detach().cpu()
    return out


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np

    tsp_data = {
        64: "/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/TSP Instances/coordinates_N64.txt",
        128: "/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/TSP Instances/coordinates_N128.txt",
        256: "/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/TSP Instances/coordinates_N256.txt",
    }

    seeds = [111, 112, 113]
    n_anneal_grid = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    for i in [64, 128, 256]:
        # ---- load coordinates ----
        coords = []
        with open(tsp_data[i], "r") as f:
            lines = f.readlines()
            n = int(lines[0].strip())
            for line in lines[1:]:
                x, y = map(float, line.strip().split())
                coords.append([x, y])

        coordinates = torch.tensor(coords, dtype=torch.float32, device=torch.device("cpu"))
        env = TSP(coordinates)

        # Collect per-chain best energies (E_best, shape (B,)) across seeds for each n_anneal
        stats = {na: [] for na in n_anneal_grid}

        for n_anneal in n_anneal_grid:
            for seed in seeds:
                result = simulated_annealing_tsp(
                    env,
                    batch_size=128,
                    warmup_steps=2000,
                    n_temps=n_anneal,          # use n_anneal as temperature ladder length
                    steps_per_temp=5,
                    T0=2.0,
                    Tend=1e-3,
                    seed=seed,                 # use actual loop seed
                    device=torch.device("cpu"),
                    return_traces=False,       # do not compute/return trace_best
                )

                # Save per-run best tours (per-chain)
                out_dir = Path(f"runs/sa_tsp_{i}New/seed{seed}2/n_anneal{n_anneal}")
                out_dir.mkdir(parents=True, exist_ok=True)
                best_idx = result["best_tour_index"].detach().cpu()
                torch.save(best_idx, out_dir / "best_tour_index.pt")

                # Accumulate per-chain best energies for summary
                e_best = result["best_energy"].detach().cpu().numpy().astype(float)  # shape (B,)
                stats[n_anneal].append(e_best)

        # ---- write a single summary file per N ----
        summary_dir = Path(f"runs/sa_tsp_{i}New")
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.csv"

        with open(summary_path, "w") as f:
            f.write("n_anneal,min,mean\n")
            for na in sorted(stats.keys()):
                # Pool all chains across all seeds for this n_anneal
                all_e = np.concatenate(stats[na], axis=0)  # shape (num_seeds * B,)
                f.write(f"{na},{all_e.min():.9f},{all_e.mean():.9f}\n")

        print(f"Wrote summary to {summary_path}")

                