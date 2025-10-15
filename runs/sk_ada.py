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
import math

current_dir = os.path.dirname(os.path.abspath(__file__))

from envs import SherringtonKirkpatrick
from models import DilatedRNN
from neighborModels import MLP as NeighborMLP
from supervisedLosses import *

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_



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

    T0 = config["T0"]

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
    ##Algorithm Here
    def sample_and_logp(model, batch_size: int):
        """
        Returns:
            solutions: LongTensor [B, L or ...]
            logp: FloatTensor [B]   (log probability under current model)
        Tries several method names for robustness.
        """
        out = model.sample(batch_size)
        logp = model(out)
        logp = logp.reshape(-1).to(torch.float32)
        return out, logp

    # ---- helpers: SNIS estimation for E_{π_T}[H] and Var_{π_T}(H) ----
    @torch.no_grad()
    def snis_piT_stats(energies, logp, T, w_clip=1e6):
        """
        energies: FloatTensor [B]
        logp:     FloatTensor [B]  (log prob under p_λ)
        Returns:
            E_pi_H_hat, Var_pi_H_hat, ESS, weights (normalized)
        """
        beta = 1.0 / max(T, 1e-12)
        # log weights: log w_i ∝ -β H_i - log p(s_i)
        logw = (-beta * energies.to(torch.float32) - logp).detach()
        logw = logw - torch.max(logw)  # stabilize
        w = torch.exp(logw)
        if math.isfinite(w_clip):
            w = torch.clamp(w, max=w_clip)
        sum_w = torch.sum(w) + 1e-20
        w_norm = w / sum_w

        # ESS (standard)
        ess = (sum_w ** 2) / (torch.sum(w ** 2) + 1e-20)

        # Weighted stats under π_T
        E_pi_H = torch.sum(w_norm * energies.to(torch.float32))
        var_pi_H = torch.sum(w_norm * (energies.to(torch.float32) - E_pi_H) ** 2)
        return E_pi_H.item(), var_pi_H.item(), ess.item(), w_norm

    # ---- free-energy MC estimate under current model ----
    @torch.no_grad()
    def estimate_free_energy(energies, logp, T):
        # F(λ;T) = E_p[H] + T E_p[log p]
        return (energies.mean().item() + T * logp.mean().item())

    # ---- config knobs (with robust defaults) ----
    T0 = float(config.get("T0", 10.0))
    T_min = float(config.get("T_min", 1e-3))
    delta_kl = float(config.get("delta_kl", 1e-2))
    schedule_mode = str(config.get("schedule_mode", "fo+so")).lower()  # "fo" or "fo+so"
    inner_steps = int(config.get("inner_steps", 10))
    est_batch_size = int(config.get("est_batch_size", config["batch_size"]))
    w_clip = float(config.get("w_clip", 1e6))
    ess_warn_frac = float(config.get("ess_warn_frac", 0.10))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    log_every = int(config.get("log_every", 1))

    eps_E = 1e-12
    eps_zeta = 1e-12

    # Move model to device
    model = model.to(device)
    model.train()

    # Initial temperature
    T = float(T0)
    n_outer = int(config["n_anneal"])
    B_est = est_batch_size
    B_train = int(config["batch_size"])

    # For logging
    print("=== KL-controlled VCA (IG-2) start ===")
    print(f"[cfg] T0={T0:.6g}  T_min={T_min:.6g}  delta_kl={delta_kl:.3e}  "
          f"mode={schedule_mode}  inner_steps={inner_steps}  "
          f"w_clip={w_clip}  B_est={B_est}  B_train={B_train}")

    # Optional moving baseline to reduce variance across inner steps
    moving_baseline = None
    baseline_momentum = 0.9

    # Outer annealing loop
    for k in range(n_outer):
        if T <= T_min + 1e-12:
            print(f"[{k:04d}] Reached T_min={T_min:.3e}. Stopping anneal.")
            break

        # --------- Estimation pass (no grad) ---------
        with torch.no_grad():
            # Sample from current model
            est_solutions, est_logp = sample_and_logp(model, B_est)
            est_energies = env.energy(est_solutions).to(torch.float32)  # [B]
            mean_E_p = est_energies.mean().item()
            min_E_p = est_energies.min().item()
            max_E_p = est_energies.max().item()
            entropy_est = -(est_logp.mean().item())  # S ≈ -E_p[log p]
            F_pre = estimate_free_energy(est_energies, est_logp, T)  # F(λ;T)

            # SNIS to estimate E_{π_T}[H] and Var_{π_T}(H)
            E_pi_H_hat, Var_pi_H_hat, ESS, w_norm = snis_piT_stats(
                est_energies, est_logp, T, w_clip=w_clip
            )
            DeltaE_hat = max(mean_E_p - E_pi_H_hat, 0.0)  # clamp small negatives to 0
            zeta_hat = Var_pi_H_hat / max(T ** 4, 1e-24)

        # --------- Step proposals (FO and SO) ---------
        # First-order (controls de/dT term): ΔT_FO = - δ T^2 / ΔE
        if DeltaE_hat <= eps_E:
            # fallback: essentially infinite FO magnitude; let SO dominate
            DeltaT_FO = -float("inf")
        else:
            DeltaT_FO = - delta_kl * (T ** 2) / (DeltaE_hat + 1e-20)

        # Second-order (thermo-length near-equilibrium): ΔT_SO = - sqrt(2 δ / ζ)
        if zeta_hat <= eps_zeta:
            DeltaT_SO = -float("inf")
        else:
            DeltaT_SO = - math.sqrt(2.0 * delta_kl / (zeta_hat + 1e-20))

        # Choose conservative step
        if schedule_mode == "fo":
            chosen_DeltaT = DeltaT_FO
        else:
            # both negative; take the less negative (closer to zero) to satisfy both budgets
            chosen_DeltaT = max(DeltaT_FO, DeltaT_SO)

        # Guardrails
        if not math.isfinite(chosen_DeltaT) or chosen_DeltaT > 0.0:
            # If estimates are degenerate, take a tiny step to avoid stalling
            chosen_DeltaT = -min(1e-3, 0.05 * max(T, 1e-6))
        # Do not overshoot below T_min
        if T + chosen_DeltaT < T_min:
            chosen_DeltaT = T_min - T

        T_next = T + chosen_DeltaT
        beta = 1.0 / max(T, 1e-12)

        # --------- Inner re-equilibration at T_next (with grads) ---------
        # We minimize F(λ;T_next) using REINFORCE-style gradient:
        #   ∇ F = E_p[ ∇ log p * ( H + T * log p ) ]
        # Use baseline b (moving average) to reduce variance (doesn't change expectation).
        model.train()
        pre_F = F_pre  # for logging
        for t in range(inner_steps):
            solutions, logp = sample_and_logp(model, B_train)
            energies = env.energy(solutions).to(torch.float32)  # [B]
            # baseline on (H + T log p)
            hv = (energies + T_next * logp).detach()
            batch_mean_hv = hv.mean()
            if moving_baseline is None:
                moving_baseline = batch_mean_hv.item()
            else:
                moving_baseline = (
                    baseline_momentum * moving_baseline
                    + (1.0 - baseline_momentum) * batch_mean_hv.item()
                )
            b = torch.tensor(moving_baseline, device=energies.device, dtype=energies.dtype)

            # Surrogate loss whose gradient equals ∇ F (up to baseline):
            #   loss = E[ (H + T log p - b) * log p ]
            # Minimizing this does gradient descent on F.
            advantage = (hv - b)  # detached already
            loss = torch.mean(advantage * logp)  # logp has grads

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # gradient health check + clipping
            total_norm = 0.0
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = math.sqrt(total_norm)
            if not math.isfinite(total_norm):
                print(f"[WARN] NaN/Inf gradient norm at outer {k}, inner {t}; skipping step.")
            else:
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

        # Post inner update diagnostics at T_next
        with torch.no_grad():
            diag_solutions, diag_logp = sample_and_logp(model, B_est)
            diag_energies = env.energy(diag_solutions).to(torch.float32)
            F_post = estimate_free_energy(diag_energies, diag_logp, T_next)
            mean_E_post = diag_energies.mean().item()
            min_E_post = diag_energies.min().item()
            entropy_post = -(diag_logp.mean().item())

        # --------- Logging ---------
        if (k % log_every) == 0:
            ess_frac = ESS / max(B_est, 1)
            print(
                f"[{k:04d}] T={T:.5g} -> {T_next:.5g}  ΔT={chosen_DeltaT:.3e}  "
                f"(FO={DeltaT_FO:.3e}, SO={DeltaT_SO:.3e}) | "
                f"E_p: mean={mean_E_p:.6g} min={min_E_p:.6g} max={max_E_p:.6g} | "
                f"E_π̂={E_pi_H_hat:.6g}  ΔÊ={DeltaE_hat:.6g}  Var_π̂={Var_pi_H_hat:.6g}  ζ̂={zeta_hat:.6g} | "
                f"ESS={ESS:.1f} ({ess_frac*100:.1f}%)  S≈{entropy_est:.6g} -> {entropy_post:.6g} | "
                f"F(λ;T): {pre_F:.6g} -> F(λ;T_next): {F_post:.6g} | "
                f"grad_norm≈{total_norm:.3g}"
            )
            if ess_frac < ess_warn_frac:
                print(
                    f"[WARN] Low ESS at step {k}: ESS={ESS:.1f} ({ess_frac*100:.1f}% of {B_est}). "
                    f"Consider increasing est_batch_size or relaxing w_clip."
                )
            if not (math.isfinite(DeltaT_FO) or math.isfinite(DeltaT_SO)):
                print(
                    f"[WARN] Both FO and SO steps non-finite at step {k}. "
                    f"Falling back to tiny ΔT; check w_clip/estimation noise."
                )

        # Advance temperature
        T = T_next

    print("=== KL-controlled VCA (IG-2) done ===")
    model.eval()

            
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
    for n_anneal in [8192]:
        config["n_anneal"] = n_anneal
        result = main(config, sk_data)
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([n_anneal, result['min'], result['mean']])

        print(f"Saved results to {out_path}")
