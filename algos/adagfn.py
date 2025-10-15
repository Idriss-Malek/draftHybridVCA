from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
from torch import nn, optim

from .adavca import snis_piT_stats, estimate_free_energy
from envs.env import Env


def _trajectory_balance_loss(
    forward_logprob: torch.Tensor,
    backward_logprob: torch.Tensor,
    terminal_log_flow: torch.Tensor,
    log_reward: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    forward_sum = forward_logprob.sum(dim=-1)
    backward_sum = backward_logprob.sum(dim=-1)
    balance = beta * terminal_log_flow + forward_sum - backward_sum - log_reward.view(-1)
    return torch.mean(balance ** 2)


def adaptive_gflownet(
    env: Env,
    policy: nn.Module,
    *,
    optimizer: optim.Optimizer,
    config: Dict[str, float],
    n_steps: int,
    batch_size: int,
    est_batch_size: Optional[int] = None,
    logZ: Optional[nn.Parameter] = None,
    scheduler: Optional[Callable[[optim.Optimizer, int], None]] = None,
    device: Optional[torch.device] = None,
    sample_fn: Optional[Callable[[nn.Module, int], torch.Tensor]] = None,
    forward_logprob_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    backward_logprob_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    warmup_steps: int = 0,
    anneal_steps: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if device is None:
        device = torch.device("cpu")

    policy = policy.to(device)

    if logZ is None:
        logZ = nn.Parameter(torch.tensor(0.0, device=device))
        optimizer.add_param_group({"params": [logZ]})

    if sample_fn is None:
        sample_fn = lambda model, B: model.sample(B)

    if forward_logprob_fn is None:
        forward_logprob_fn = lambda model, x: model(x)

    if backward_logprob_fn is None:
        backward_logprob_fn = lambda model, x: torch.zeros_like(
            model(x).unsqueeze(-1).repeat(1, x.shape[1])
        )

    T = float(config.get("T0", 10.0))
    T_min = float(config.get("T_min", 1e-3))
    delta_kl = float(config.get("delta_kl", 1e-2))
    schedule_mode = str(config.get("schedule_mode", "fo+so")).lower()
    inner_steps = int(config.get("inner_steps", 10))
    est_batch_size = int(est_batch_size or config.get("est_batch_size", batch_size))
    w_clip = float(config.get("w_clip", 1e6))
    eps_E = 1e-12
    eps_zeta = 1e-12

    loss_history = []
    reward_stats = []
    temperature_history = []
    delta_history = []
    energy_means = []
    energy_mins = []
    energy_maxs = []
    energy_vars = []

    total_steps = warmup_steps + (anneal_steps if anneal_steps is not None else 0)
    if total_steps <= 0:
        total_steps = n_steps
    else:
        n_steps = total_steps

    def record_stats(energies: torch.Tensor, temperature_value: float, delta_value: float) -> None:
        temperature_history.append(temperature_value)
        delta_history.append(delta_value)
        energy_means.append(energies.mean().item())
        energy_mins.append(energies.min().item())
        energy_maxs.append(energies.max().item())
        energy_vars.append(energies.var(unbiased=False).item())

    def train_iteration(current_T: float, next_T: float, step_index: int) -> float:
        policy.train()
        optimizer.zero_grad()

        trajectories = sample_fn(policy, batch_size).to(device)
        forward_logprob = forward_logprob_fn(policy, trajectories)
        backward_logprob = backward_logprob_fn(policy, trajectories)

        energies = env.energy(trajectories)
        rewards = torch.exp(-energies / max(next_T, torch.finfo(energies.dtype).tiny))
        log_reward = torch.log(rewards + 1e-12)

        loss = _trajectory_balance_loss(
            forward_logprob,
            backward_logprob,
            logZ,
            log_reward,
            beta=1.0,
        )

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler(optimizer, step_index)

        loss_history.append(loss.item())
        reward_stats.append(rewards.mean().item())
        record_stats(energies, current_T, next_T - current_T)

        return next_T

    step_index = 0

    # Warm-up phase (fixed temperature)
    for _ in range(warmup_steps):
        T = train_iteration(T, T, step_index)
        step_index += 1

    # Adaptive annealing phase
    for _ in range(step_index, n_steps):
        with torch.no_grad():
            est_traj = sample_fn(policy, est_batch_size)
            est_traj = est_traj.to(device)
            est_logp = forward_logprob_fn(policy, est_traj)
            est_energy = env.energy(est_traj).to(torch.float32)
            mean_E = est_energy.mean().item()

            E_pi_h, Var_pi_h, _, _ = snis_piT_stats(est_energy, est_logp.sum(dim=-1), T, w_clip=w_clip)
            DeltaE_hat = max(mean_E - E_pi_h, 0.0)
            zeta_hat = Var_pi_h / max(T ** 4, 1e-24)

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
                chosen_DeltaT = -T_min
            else:
                chosen_DeltaT = T_min - T

        T = train_iteration(T, T + chosen_DeltaT, step_index)
        step_index += 1

    return {
        "logZ": logZ.detach(),
        "temperature": torch.tensor(temperature_history, dtype=torch.float32),
        "delta": torch.tensor(delta_history, dtype=torch.float32),
        "loss_history": torch.tensor(loss_history, dtype=torch.float32),
        "reward_stats": torch.tensor(reward_stats, dtype=torch.float32),
        "energy_mean": torch.tensor(energy_means, dtype=torch.float32),
        "energy_min": torch.tensor(energy_mins, dtype=torch.float32),
        "energy_max": torch.tensor(energy_maxs, dtype=torch.float32),
        "energy_var": torch.tensor(energy_vars, dtype=torch.float32),
    }
