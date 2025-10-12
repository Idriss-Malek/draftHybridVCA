import torch
import math
import os
import csv
from torch.nn.utils import clip_grad_norm_

def sample_and_logp(model, batch_size: int):
    out = model.sample(batch_size)
    logp = model(out)
    logp = logp.reshape(-1).to(torch.float32)
    return out, logp

@torch.no_grad()
def snis_piT_stats(energies, logp, T, w_clip=1e6):
    beta = 1.0 / max(T, 1e-12)
    logw = (-beta * energies.to(torch.float32) - logp).detach()
    logw = logw - torch.max(logw)
    w = torch.exp(logw)
    if math.isfinite(w_clip):
        w = torch.clamp(w, max=w_clip)
    sum_w = torch.sum(w) + 1e-20
    w_norm = w / sum_w
    ess = (sum_w ** 2) / (torch.sum(w ** 2) + 1e-20)
    E_pi_H = torch.sum(w_norm * energies.to(torch.float32))
    var_pi_H = torch.sum(w_norm * (energies.to(torch.float32) - E_pi_H) ** 2)
    return E_pi_H.item(), var_pi_H.item(), ess.item(), w_norm

@torch.no_grad()
def estimate_free_energy(energies, logp, T):
    return (energies.mean().item() + T * logp.mean().item())

def adaVCA(model, env, config, optimizer, record=None):
    # --- CSV setup ---
    if record is not None:
        os.makedirs(os.path.dirname(record) or ".", exist_ok=True)
        is_new = not os.path.exists(record)
        with open(record, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["iteration","Temperature","DeltaT","minE","maxE","meanE","varE","ESS"])

    T0 = float(config.get("T0", 10.0))
    T_min = float(config.get("T_min", 1e-3))
    delta_kl = float(config.get("delta_kl", 1e-2))
    schedule_mode = str(config.get("schedule_mode", "fo+so")).lower()
    inner_steps = int(config.get("inner_steps", 10))
    est_batch_size = int(config.get("est_batch_size", config["batch_size"]))
    w_clip = float(config.get("w_clip", 1e6))
    ess_warn_frac = float(config.get("ess_warn_frac", 0.10))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    device = config["device"]

    eps_E = 1e-12
    eps_zeta = 1e-12

    model = model.to(device)
    model.train()

    T = float(T0)
    n_outer = int(config["n_anneal"])
    B_est = est_batch_size
    B_train = int(config["batch_size"])

    moving_baseline = None
    baseline_momentum = 0.

    for k in range(n_outer):
        if T <= T_min:
            break
        with torch.no_grad():
            est_solutions, est_logp = sample_and_logp(model, B_est)
            est_energies = env.energy(est_solutions).to(torch.float32)
            mean_E_p = est_energies.mean().item()
            min_E_p = est_energies.min().item()
            max_E_p = est_energies.max().item()
            var_E_p = est_energies.var(unbiased=False).item()
            F_pre = estimate_free_energy(est_energies, est_logp, T)

            E_pi_H_hat, Var_pi_H_hat, ESS, w_norm = snis_piT_stats(
                est_energies, est_logp, T, w_clip=w_clip
            )
            DeltaE_hat = max(mean_E_p - E_pi_H_hat, 0.0)
            zeta_hat = Var_pi_H_hat / max(T ** 4, 1e-24)

        if DeltaE_hat <= eps_E:
            DeltaT_FO = -float("inf")
        else:
            DeltaT_FO = - delta_kl * (T ** 2) / (DeltaE_hat + 1e-20)

        if zeta_hat <= eps_zeta:
            DeltaT_SO = -float("inf")
        else:
            DeltaT_SO = - math.sqrt(2.0 * delta_kl / (zeta_hat + 1e-20))

        if schedule_mode == "fo":
            chosen_DeltaT = DeltaT_FO
        else:
            chosen_DeltaT = max(DeltaT_FO, DeltaT_SO)

        if not math.isfinite(chosen_DeltaT) or chosen_DeltaT > 0.0:
            chosen_DeltaT = -min(1e-3, 0.05 * max(T, 1e-6))
        if T + chosen_DeltaT < T_min:
            if T > 5 * T_min:
                chosen_DeltaT = - T_min
            else:
                chosen_DeltaT = T_min - T

        T_next = T + chosen_DeltaT

        model.train()
        pre_F = F_pre
        for t in range(inner_steps):
            solutions, logp = sample_and_logp(model, B_train)
            energies = env.energy(solutions).to(torch.float32)
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

            advantage = (hv - b)
            loss = torch.mean(advantage * logp)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            total_norm = 0.0
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = math.sqrt(total_norm)
            if math.isfinite(total_norm):
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

        with torch.no_grad():
            diag_solutions, diag_logp = sample_and_logp(model, B_est)
            diag_energies = env.energy(diag_solutions).to(torch.float32)
            F_post = estimate_free_energy(diag_energies, diag_logp, T_next)

        # --- append one CSV row per outer iteration ---
        if record is not None:
            with open(record, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    k,                # iteration
                    T,                # Temperature (pre-step)
                    chosen_DeltaT,    # ΔT
                    min_E_p,          # minE_p
                    max_E_p,          # maxE_p
                    mean_E_p,         # meanE_p
                    var_E_p,          # varE_p
                    ESS               # ESS
                ])

        T = T_next
