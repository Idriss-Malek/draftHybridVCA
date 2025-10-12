import sys
import os
import argparse
import json
import csv
from datetime import datetime

sys.path.append("/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA")
current_dir = os.path.dirname(os.path.abspath(__file__))

from envs import TSP
from models import DilatedRNN, VanillaRNN
from supervisedLosses import *

import torch
import torch.optim as optim
import torch.nn.functional as F


# ----------------------------- data paths -----------------------------
tsp_data = {
    64: "/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/TSP Instances/coordinates_N64.txt",
    128: "/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/TSP Instances/coordinates_N128.txt",
    256: "/home/idrissm/projects/def-mh541-ab/idrissm/neighborVCA/data/TSP Instances/coordinates_N256.txt",
}


# ------------------------- small utilities ---------------------------
def safe_device(name: str) -> torch.device:
    """Map ambiguous device strings to a valid torch.device."""
    if name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = str(name).lower()
    if n in ("cuda", "gpu", "cuda:0"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n in ("cpu",):
        return torch.device("cpu")
    # Handle common typos like "coda"
    if "cod" in n or "cda" in n:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_coords(path, device, dtype=torch.float64):
    coords = []
    with open(path, "r") as f:
        lines = f.readlines()
        _n = int(lines[0].strip())  # first line = number of coordinates
        for line in lines[1:]:
            x, y = map(float, line.strip().split())
            coords.append([x, y])
    return torch.tensor(coords, dtype=dtype, device=device)


def params_of(model):
    return [p for p in model.parameters() if p.requires_grad]


def zeros_like_params(params):
    return [torch.zeros_like(p) for p in params]


def add_inplace(params, upd):
    for p, u in zip(params, upd):
        p.data.add_(u)


def scale_inplace(vec, alpha):
    for i in range(len(vec)):
        if vec[i] is not None:
            vec[i] = vec[i] * alpha


def invC_times(vec, fisher_diag, eps):
    out = []
    for v, d in zip(vec, fisher_diag):
        if v is None:
            out.append(None)
        else:
            out.append(v / (d + eps))
    return out


def quad_form_diagC(vec, fisher_diag):
    s = 0.0
    for v, d in zip(vec, fisher_diag):
        if v is None or d is None:
            continue
        s = s + (v.detach() * d.detach() * v.detach()).sum()
    return s


def normC(vec, fisher_diag):
    val = quad_form_diagC(vec, fisher_diag)
    return torch.sqrt(val + 1e-20)


def dot_paramlists(u, v):
    s = 0.0
    for a, b in zip(u, v):
        if a is None or b is None:
            continue
        s = s + (a.detach() * b.detach()).sum()
    return s


def per_sample_score_sq(logp_scalar, params):
    """Return squared score for a single sample via grad of -log p."""
    lj = -logp_scalar
    gj = torch.autograd.grad(lj, params, retain_graph=True, allow_unused=True)
    out = []
    for g in gj:
        if g is None:
            out.append(None)
        else:
            out.append(g * g)
    return out


def ess_from_weights(wtilde):
    # ESS = 1 / sum_i w_i^2
    denom = (wtilde * wtilde).sum().item()
    return float(1.0 / max(denom, 1e-12))


def free_energy_from_samples(energies, logp, beta):
    # F_lambda(beta) = E_p[H] + (1/beta) E_p[log p]
    e_mean = energies.mean().item()
    lp_mean = logp.mean().item()
    if beta <= 1e-12:
        return float("nan")
    return e_mean + (lp_mean / beta)


# ------------------------------ main ---------------------------------
def main(config, step_csv_writer=None, print_every=1):
    # ---- resolve device (robust to "coda") ----
    device = safe_device(config.get("device"))
    record = {}

    # ---- load coordinates ----
    path = tsp_data[config["seq_size"]]
    coordinates = read_coords(path, device=device, dtype=torch.float64)

    # ---- env + model ----
    env = TSP(coordinates)
    if config["model_type"] == "DilatedRNN":
        model = DilatedRNN(config, env)
    elif config["model_type"] == "VanillaRNN":
        model = VanillaRNN(config, env)
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    # MOVE MODEL TO DEVICE (important)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]))

    # ------------------ CGTR hyperparameters (safe defaults) ------------------
    # (You can tune these per your JSON; shown defaults are sensible.)
    beta = float(config.get("beta0", 0.5))                   # initial inverse temperature
    delta = float(config.get("cgtr_delta", 0.5))             # trust-region radius (Fisher distance)
    rho   = float(config.get("cgtr_rho", 0.5))               # cap on natural-norm of Δλ
    lr_nat = float(config.get("lr_nat", 1.0))                # step-size in the natural direction
    fisher_subsample = int(config.get("fisher_subsample", 128))  # #samples for Fisher diag
    cgtr_mode = str(config.get("cgtr_mode", "simultaneous")).lower()  # "successive" or "simultaneous"
    fisher_ridge = float(config.get("fisher_ridge", 1e-6))   # damping for Fisher diag
    grad_clip = float(config.get("grad_clip", 10.0))         # optional grad clip on Δλ in Euclidean norm (after nat-scaling)
    dtype_params = next(p for p in model.parameters() if p.requires_grad).dtype

    model.train()
    params = params_of(model)

    # ----------------------------- annealing loop -----------------------------
    for k in range(int(config["n_anneal"])):
        # 1) Sample from the current model
        with torch.no_grad():
            solutions = model.sample(int(config["batch_size"]))  # shape [B, ...]
        logp = model(solutions).to(dtype_params)                 # shape [B], differentiable
        energies = env.energy(solutions).to(dtype_params)        # shape [B], differentiable wrt λ only through logp (not needed)

        # (Free energy before update — for inspection)
        F_old = free_energy_from_samples(energies, logp, beta)

        # 2) Importance weights for q_beta (FIXED: NO -logp TERM!)
        #    w ∝ exp(-beta * H), since samples are from p_lambda
        with torch.no_grad():
            logw = (-beta * energies).detach()
            logw = logw - logw.max()  # numerical stability
            w = torch.exp(logw)
            wsum = w.sum() + 1e-12
            wtilde = w / wsum                     # normalized weights, Σ w̃ = 1
            Ew = (wtilde * energies).sum()        # E_{q_β}[H] via IS (normalized)
            ESS = ess_from_weights(wtilde)

        # 3) IS-corrected B = -Cov_{q_β}(H, s)
        #    Gradient of Σ_i w̃_i (H_i - Ew) * (−log p_λ(σ_i))
        loss_energy_IS = ((wtilde * (energies - Ew)).detach() * (-logp)).sum()
        ge = torch.autograd.grad(loss_energy_IS, params, retain_graph=True, allow_unused=True)
        ge = [g if g is not None else torch.zeros_like(p) for g, p in zip(ge, params)]
        B_vec = [g.detach() for g in ge]  # B

        # 4) Entropy gradient piece: (1/β) E_p[ (log p) s ]
        #    Use detached multiplier trick for unbiased score estimator.
        if beta <= 1e-12:
            gl = [torch.zeros_like(p) for p in params]
        else:
            loss_entropy = -(1.0 / beta) * (logp.detach() * logp).mean()
            gl = torch.autograd.grad(loss_entropy, params, retain_graph=True, allow_unused=True)
            gl = [g if g is not None else torch.zeros_like(p) for g, p in zip(gl, params)]

        # 5) Full gradient of F_λ(β): g = -ge - gl
        g_vec = [-(ge_i + gl_i) for ge_i, gl_i in zip(ge, gl)]

        # 6) Empirical Fisher diagonal C ≈ E[s ⊙ s]
        m = min(int(logp.shape[0]), fisher_subsample)
        idx = torch.randint(low=0, high=logp.shape[0], size=(m,), device=logp.device)
        fisher_diag = zeros_like_params(params)
        for j in idx:
            ssq = per_sample_score_sq(logp[j], params)
            for t, g2 in enumerate(ssq):
                if g2 is None:
                    continue
                fisher_diag[t] += g2
        for t in range(len(fisher_diag)):
            fisher_diag[t] = fisher_diag[t] / max(1, m) + fisher_ridge

        # 7) a = Var_{q_β}[H] (IS corrected with normalized weights)
        a = (wtilde * (energies - Ew) ** 2).sum().item()

        # 8) Quadratic forms used by both modes (we log them even if unused)
        #    gCinv_g = g^T C^{-1} g,   BCinvB = B^T C^{-1} B
        gCinv_g = 0.0
        BCinvB  = 0.0
        for g_i, B_i, d in zip(g_vec, B_vec, fisher_diag):
            gCinv_g += ((g_i.detach()**2) / d.detach()).sum()
            BCinvB  += ((B_i.detach()**2) / d.detach()).sum()
        gCinv_g = gCinv_g.item()
        BCinvB  = BCinvB.item()

        # 9) Take a CGTR step
        if cgtr_mode.startswith("succ"):  # ------- Successive (CGTR-S) -------
            # natural step Δλ = − lr_nat * C^{-1} g
            dlam = invC_times(g_vec, fisher_diag, fisher_ridge)
            scale_inplace(dlam, -lr_nat)

            # optional Euclidean grad clip on Δλ to avoid blowups (safe)
            if grad_clip > 0:
                # flatten and clip by global norm
                flat = torch.cat([v.view(-1) for v in dlam if v is not None])
                norm = torch.norm(flat, p=2).item()
                if norm > grad_clip:
                    scale_inplace(dlam, grad_clip / (norm + 1e-12))

            # cap natural norm ||Δλ||_C ≤ ρ
            nat_norm = normC(dlam, fisher_diag).item()
            if nat_norm > rho:
                scale_inplace(dlam, rho / max(nat_norm, 1e-12))
                nat_norm = rho  # after scaling, equals cap

            # scalars for β-step
            quad = quad_form_diagC(dlam, fisher_diag).item()  # Δλ^T C Δλ
            b = dot_paramlists(B_vec, dlam).item()             # B^T Δλ

            # Δβ from trust-region constraint  (disc >= 0 for feasible step)
            # disc = b^2 + a * max(delta^2 - quad, 0)
            slack = max(delta * delta - quad, 0.0)
            disc = b * b + a * slack
            d_beta = 0.0 if a <= 1e-18 else ((-b + (disc ** 0.5)) / a)

        else:  # ----------------------- Simultaneous (CGTR-J) ---------------
            num = max(delta * delta - gCinv_g, 0.0)
            den = max(a - BCinvB, 1e-18)

            if num <= 0.0 or den <= 0.0:
                # fallback: limited β step + small natural step
                d_beta = delta / (a ** 0.5 + 1e-18) if a > 1e-18 else 0.0
                dlam = invC_times(g_vec, fisher_diag, fisher_ridge)
                scale_inplace(dlam, -min(lr_nat, 0.25))
            else:
                d_beta = (num / den) ** 0.5
                # Δλ = − C^{-1}( g + B Δβ )
                g_plus = [g_i + B_i * d_beta for g_i, B_i in zip(g_vec, B_vec)]
                dlam = invC_times(g_plus, fisher_diag, fisher_ridge)
                scale_inplace(dlam, -1.0)

                # optional cap on ||Δλ||_C
                nat_norm = normC(dlam, fisher_diag).item()
                if nat_norm > rho:
                    scale_inplace(dlam, rho / max(nat_norm, 1e-12))
                    nat_norm = rho

            quad = quad_form_diagC(dlam, fisher_diag).item()
            b = dot_paramlists(B_vec, dlam).item()
            disc = (b * b + a * max(delta * delta - quad, 0.0))
            slack = max(delta * delta - quad, 0.0)
            num = max(delta * delta - gCinv_g, 0.0)
            den = max(a - BCinvB, 1e-18)

        # 10) Apply updates (λ and β)
        add_inplace(params, dlam)
        beta_before = beta
        beta = beta + float(d_beta)

        # 11) Diagnostics after update: F_new on the SAME minibatch (off-policy but informative)
        with torch.no_grad():
            logp_new = model(solutions)  # reuse the same solutions
        F_new = free_energy_from_samples(energies, logp_new, beta)
        dF = (F_new - F_old) if (F_new == F_new and F_old == F_old) else float("nan")  # protect NaNs

        # 12) Print + CSV log
        mode_str = "successive" if cgtr_mode.startswith("succ") else "simultaneous"
        row = {
            "step": k,
            "mode": mode_str,
            "beta_before": beta_before,
            "delta": delta,
            "delta_eff": delta,  # constant here; adapt externally if you want a schedule
            "d_beta": float(d_beta),
            "ESS": ESS,
            "a": a,
            "nat_norm": nat_norm,
            "E_mean": energies.mean().item(),
            "F_old": F_old,
            "F_new": F_new,
            "dF": dF,
            "b_BtDeltaLam": b,
            "quad_DlC_Dl": quad,
            "disc": disc,
            "gCinv_g": gCinv_g,
            "BCinvB": BCinvB,
            "num": (delta * delta - gCinv_g) if not cgtr_mode.startswith("succ") else float("nan"),
            "den": (a - BCinvB) if not cgtr_mode.startswith("succ") else float("nan"),
        }

        if step_csv_writer is not None:
            step_csv_writer.writerow([row[c] for c in [
                "step","mode","beta_before","delta","delta_eff","d_beta","ESS","a","nat_norm",
                "E_mean","F_old","F_new","dF","b_BtDeltaLam","quad_DlC_Dl","disc",
                "gCinv_g","BCinvB","num","den"
            ]])

        if (k % max(1, int(print_every)) == 0):
            print(
                f"step={row['step']} mode={row['mode']} "
                f"β→{beta:.6f} (Δβ={row['d_beta']:.6f})  "
                f"ESS={row['ESS']:.1f}  a={row['a']:.4f}  "
                f"||Δλ||_C={row['nat_norm']:.4f}  Ē={row['E_mean']:.4f}  "
                f"F_old={row['F_old']:.4f} F_new={row['F_new']:.4f} ΔF={row['dF']:.4f}  "
                f"b={row['b_BtDeltaLam']:.4f}  quad={row['quad_DlC_Dl']:.4f}  disc={row['disc']:.4f}  "
                f"gCinv_g={row['gCinv_g']:.4f} BCinvB={row['BCinvB']:.4f}"
            )

    # ----------------------------- evaluation -----------------------------
    record["mean"] = 0.0
    record["min"] = float("inf")
    with torch.no_grad():
        for st in range(1000):
            solutions = model.sample(int(config["batch_size"]))
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
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Print diagnostics every this many steps.",
    )
    args = parser.parse_args()

    # Load config JSON -> dict
    with open(args.config, "r") as f:
        config = json.load(f)

    # Where to put final sweep results + per-step logs
    out_dir = os.path.join(current_dir, "results", "tsp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        out_dir,
        f"tsp{config['seq_size']}_{os.path.basename(args.config)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    # Global CSV: (n_anneal, min, mean)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_anneal", "min", "mean", "per_step_log"])

    # Run sweep
    for n_anneal in [16, 512, 1024, 2048, 4096]:
        cfg = dict(config)  # copy
        cfg["n_anneal"] = 2000 + n_anneal * 5

        # per-step log path
        step_path = os.path.join(
            out_dir,
            f"steps_tsp{cfg['seq_size']}_{os.path.basename(args.config)}_na{cfg['n_anneal']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with open(step_path, "w", newline="") as sf:
            step_writer = csv.writer(sf)
            step_writer.writerow([
                "step","mode","beta_before","delta","delta_eff","d_beta","ESS","a","nat_norm",
                "E_mean","F_old","F_new","dF","b_BtDeltaLam","quad_DlC_Dl","disc",
                "gCinv_g","BCinvB","num","den"
            ])
            # Run one experiment while logging each step
            result = main(cfg, step_csv_writer=step_writer, print_every=args.print_every)

        # Append aggregate result
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([n_anneal, result["min"], result["mean"], step_path])

        print(f"[DONE] n_anneal={n_anneal}   results saved to: {out_path}\n"
              f"       per-step diagnostics: {step_path}")
