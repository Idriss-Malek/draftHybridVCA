from __future__ import annotations

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

from typing import Callable, Dict, Optional

import torch
from torch import nn, optim

from envs.env import Env


def _trajectory_balance_loss(
    forward_logprob: torch.Tensor,
    backward_logprob: torch.Tensor,
    terminal_log_flow: torch.Tensor,
    log_reward: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Implements the Trajectory Balance loss:
        (beta * log Z + sum_t log P_F - sum_t log P_B - log R(x))^2
    """
    forward_sum = forward_logprob.sum(dim=-1)
    backward_sum = backward_logprob.sum(dim=-1)
    log_reward = log_reward.view(-1)
    balance = beta * terminal_log_flow + forward_sum - backward_sum - log_reward
    return torch.mean(balance ** 2)


def train_gflownet(
    env: Env,
    policy: nn.Module,
    *,
    optimizer: optim.Optimizer,
    n_steps: int,
    batch_size: int,
    temperature: float,
    logZ: Optional[nn.Parameter] = None,
    scheduler: Optional[Callable[[optim.Optimizer, int], None]] = None,
    device: Optional[torch.device] = None,
    sample_fn: Optional[Callable[[nn.Module, int], torch.Tensor]] = None,
    forward_logprob_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    backward_logprob_fn: Optional[
        Callable[[nn.Module, torch.Tensor], torch.Tensor]
    ] = None,
    warmup_steps: int = 0,
    anneal_steps: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Train a policy network using the Trajectory Balance objective.

    Args:
        env: Environment compatible with the `Env` abstraction.
        policy: Policy network; must provide `sample(batch_size)` and forward returning log-probs.
        optimizer: Optimizer for policy parameters (and optionally logZ).
        n_steps: Number of training iterations.
        batch_size: Number of trajectories per batch.
        temperature: Fixed temperature T used in the reward definition exp(-E/T).
        logZ: Optional learnable log-partition parameter. Created if None.
        scheduler: Optional learning rate scheduler callable (optimizer, step_idx).
        device: Torch device to run computations on.
        sample_fn: Optional custom sampling function; defaults to `policy.sample`.
        forward_logprob_fn: Optional callable returning forward log probabilities.
        backward_logprob_fn: Optional callable returning backward log probabilities.

    Returns:
        Dictionary with final `logZ`, `loss_history`, and `reward_stats`.
    """
    if device is None:
        device = torch.device("cpu")

    policy = policy.to(device)

    if logZ is None:
        logZ = nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float32))
        optimizer.add_param_group({"params": [logZ]})

    if sample_fn is None:
        sample_fn = lambda model, B: model.sample(B)

    if forward_logprob_fn is None:
        forward_logprob_fn = lambda model, x: model(x)

    if backward_logprob_fn is None:
        backward_logprob_fn = lambda model, x: torch.zeros_like(
            model(x).unsqueeze(-1).repeat(1, x.shape[1])
        )

    loss_history = []
    reward_stats = []
    energy_means = []
    energy_mins = []
    energy_maxs = []
    energy_vars = []

    total_steps = warmup_steps + (anneal_steps if anneal_steps is not None else 0)
    if total_steps <= 0:
        total_steps = n_steps
    else:
        n_steps = total_steps

    for step in range(n_steps):
        policy.train()
        optimizer.zero_grad()

        trajectories = sample_fn(policy, batch_size)
        trajectories = trajectories.to(device)

        forward_logprob = forward_logprob_fn(policy, trajectories)
        backward_logprob = backward_logprob_fn(policy, trajectories)

        energies = env.energy(trajectories)
        rewards = torch.exp(-energies / temperature)
        log_reward = torch.log(rewards + 1e-12)

        energy_means.append(energies.mean().item())
        energy_mins.append(energies.min().item())
        energy_maxs.append(energies.max().item())
        energy_vars.append(energies.var(unbiased=False).item())

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
            scheduler(optimizer, step)

        loss_history.append(loss.item())
        reward_stats.append(rewards.mean().item())

    return {
        "logZ": logZ.detach(),
        "loss_history": torch.tensor(loss_history, dtype=torch.float32),
        "reward_stats": torch.tensor(reward_stats, dtype=torch.float32),
        "energy_mean": torch.tensor(energy_means, dtype=torch.float32),
        "energy_min": torch.tensor(energy_mins, dtype=torch.float32),
        "energy_max": torch.tensor(energy_maxs, dtype=torch.float32),
        "energy_var": torch.tensor(energy_vars, dtype=torch.float32),
    }
