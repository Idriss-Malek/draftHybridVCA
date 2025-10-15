import sys
from pathlib import Path
    sk_instances_dir = _repo_root / "data" / "SK_Instances"
    sk_data = str(sk_instances_dir / "100_SK_seed1.txt")

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

from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F

import sys
from envs import SherringtonKirkpatrick as SK

@torch.no_grad()
def simulated_annealing_sk(
    env,                          
    init_solution: Optional[torch.Tensor] = None,  # (B,N,2) one-hot over {-1,+1}
    batch_size: int = 128,
    warmup_steps: int = 0,
    n_temps: int = 200,
    steps_per_temp: int = 250,
    T0: float = 2.0,
    Tend: float = 1e-3,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    return_traces: bool = False,
) -> Dict[str, Any]:
    """
    Simulated annealing for Sherrington–Kirkpatrick (zero field) using single-spin flips.
    - One-hot solutions X: (B, N, 2), channels encode {-1,+1} as [1,0] and [0,1].
    - At each inner iteration, flip one random site per chain and Metropolis-accept.
    - 'warmup_steps': steps at fixed T0 (each 'step' performs N single-spin proposals).
    - Then anneal linearly from T0 to Tend across 'n_temps', with 'steps_per_temp' steps each.
    Returns per-chain bests, a global best, and an optional trace of the global best energy.
    """
    assert n_temps >= 1 and steps_per_temp >= 0 and warmup_steps >= 0

    # In SK env, dtype/device come from J
    dtype = env.J.dtype
    if device is None:
        device = env.J.device

    # RNG (deterministic if seed is provided)
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)
        rand = lambda *shape: torch.rand(*shape, device=device, generator=g)
        randint = lambda low, high, shape: torch.randint(low, high, shape, device=device, generator=g)
    else:
        rand = lambda *shape: torch.rand(*shape, device=device)
        randint = lambda low, high, shape: torch.randint(low, high, shape, device=device)

    N = env.N
    B = init_solution.shape[0] if init_solution is not None else batch_size

    # ---------- helpers ----------
    def random_onehot(B, N):
        # Uniform spins in {0,1}; channel 0 => -1, channel 1 => +1
        labels = (rand(B, N) > 0.5).long()              # (B,N) in {0,1}
        X = F.one_hot(labels, num_classes=2).to(dtype=dtype)  # (B,N,2) in {0,1}
        return X

    def spins_from_onehot(X):
        # For convenience in outputs (B,N)
        return X[..., 1] - X[..., 0]

    # ---------- initialize ----------
    if init_solution is None:
        X = random_onehot(B, N).to(device)
    else:
        assert init_solution.shape == (B, N, 2)
        X = init_solution.to(device=device, dtype=dtype)

    E = env.energy(X)           # (B,)
    X_best = X.clone()
    E_best = E.clone()

    # Global best across batch
    best_idx = torch.argmin(E)
    global_best = {
        "onehot": X[best_idx].clone(),                  # (N,2)
        "spins":  env._onehot_to_spins(X[best_idx:best_idx+1])[0].clone(),  # (N,)
        "energy": E[best_idx].item(),
    }

    total_steps = warmup_steps + n_temps * steps_per_temp
    trace_best = torch.empty(total_steps, device=device, dtype=dtype) if return_traces else None
    step_counter = 0

    # One Metropolis–Hastings proposal per chain at temperature T
    def mh_step_at_temperature(T: torch.Tensor):
        nonlocal X, E, X_best, E_best, global_best

        # Sample site index i per chain
        i = randint(0, N, (B,))  # (B,)

        # Propose single-spin flip at i
        X_prop = env.neighbor(X, i)       # (B,N,2)
        E_prop = env.energy(X_prop)       # (B,)

        dE = E_prop - E
        T_safe = torch.clamp(T, min=torch.finfo(dtype).tiny)
        accept = (dE <= 0) | (torch.log(rand(B)) < (-dE / T_safe))

        if accept.any():
            mask = accept.view(B, 1, 1)
            X = torch.where(mask, X_prop, X)
            E = torch.where(accept, E_prop, E)

        # Update per-chain and global bests
        better = E < E_best
        if better.any():
            X_best = torch.where(better.view(B, 1, 1), X, X_best)
            E_best = torch.where(better, E, E_best)

            cand_idx = torch.argmin(E)
            cand_energy = E[cand_idx].item()
            if cand_energy < global_best["energy"]:
                global_best["onehot"] = X[cand_idx].clone()
                global_best["spins"]  = env._onehot_to_spins(X[cand_idx:cand_idx+1])[0].clone()
                global_best["energy"] = cand_energy

    # ---------- warm-up at fixed T0 ----------
    T0_tensor = torch.tensor(T0, device=device, dtype=dtype)
    for _ in range(warmup_steps):
        for _ in range(N):
            mh_step_at_temperature(T0_tensor)
        if return_traces:
            trace_best[step_counter] = torch.tensor(global_best["energy"], device=device, dtype=dtype)
        step_counter += 1

    # ---------- temperature ladder (linear) ----------
    if n_temps == 1:
        temps = torch.tensor([T0], device=device, dtype=dtype)
    else:
        temps = torch.linspace(T0, Tend, steps=n_temps, device=device, dtype=dtype)
        temps = torch.clamp(temps, min=torch.finfo(dtype).tiny)

    # ---------- anneal ----------
    for Tk in temps:
        for _ in range(steps_per_temp):
            for __ in range(N):
                mh_step_at_temperature(Tk)
            if return_traces:
                trace_best[step_counter] = torch.tensor(global_best["energy"], device=device, dtype=dtype)
            step_counter += 1

    out = {
        "best_onehot": X_best,                              # (B,N,2)
        "best_spins":  env._onehot_to_spins(X_best),           # (B,N)
        "best_energy": E_best,                              # (B,)
        "global_best": global_best,                         # {"onehot": (N,2), "spins": (N,), "energy": float}
    }
    if return_traces:
        out["trace_best"] = trace_best.detach().cpu()
    return out

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


if __name__ == "__main__":
    import torch
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    
    J = read_matrix(sk_data).to(dtype=torch.float64, device=device)

    result = simulated_annealing_sk(
        SK(J),
        batch_size=128,
        warmup_steps=2000,
        n_temps=1,
        steps_per_temp=5,
        T0=2.0,
        Tend=1e-3,
        seed=0,
        device=torch.device("cuda"),
        return_traces=True,
    )

    # Print shapes (skip dicts)
    for key in result:
        try:
            print(result[key].shape)
        except:
            continue
    print(result["trace_best"])
    print(result["global_best"])

    # Save outputs (mirror your TSP script)
    out_dir = Path(f"runs/sa_sk_{J.shape[0]}N_seed3")
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = result["trace_best"].detach().cpu().numpy()
    with open(out_dir / "trace_best.csv", "w") as f:
        f.write("global_best_energy\n")
        for v in trace:
            f.write(f"{float(v)}\n")

    best_spins = result["best_spins"].detach().cpu()
    torch.save(best_spins, out_dir / "best_spins.pt")
