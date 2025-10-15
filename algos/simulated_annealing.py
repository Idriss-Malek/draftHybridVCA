from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch

from envs.env import Env

TemperatureSchedule = Union[str, Callable[[int], float]]
NeighborArgSampler = Callable[
    [Env, torch.Tensor, Optional[torch.Generator]],
    Tuple[Tuple[Any, ...], Dict[str, Any]],
]
Initializer = Callable[[Env, int, torch.device, Optional[torch.Generator]], torch.Tensor]


def _default_neighbor_args(
    env: Env,
    state: torch.Tensor,
    generator: Optional[torch.Generator],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Produce random positional arguments for `env.neighbor` based on its signature.
    Supports neighbors that take up to two additional positional index tensors.
    """
    params = list(inspect.signature(env.neighbor).parameters.values())
    if not params:
        raise ValueError("`env.neighbor` signature is empty.")

    extras = params[1:]  # skip `solution`
    if not extras:
        return (), {}

    if not hasattr(env, "N"):
        raise AttributeError(
            "Environment must define attribute `N` for the default neighbor sampler. "
            "Supply `neighbor_arg_sampler` for custom behavior."
        )
    N = getattr(env, "N")
    if not isinstance(N, int) or N <= 0:
        raise TypeError("Environment attribute `N` must be a positive integer.")

    B = state.shape[0]
    device = state.device

    def randint(high: int) -> torch.Tensor:
        if generator is None:
            return torch.randint(0, high, (B,), device=device)
        return torch.randint(0, high, (B,), device=device, generator=generator)

    if len(extras) == 1:
        idx = randint(N)
        return (idx,), {}

    if len(extras) == 2:
        if N < 2:
            raise ValueError("Need at least two elements to sample distinct indices.")
        i = randint(N)
        j = randint(N - 1)
        j = j + (j >= i)
        return (i, j), {}

    raise NotImplementedError(
        "Default neighbor sampler currently supports up to two positional index arguments. "
        "Provide `neighbor_arg_sampler` for custom signatures."
    )


def _build_temperature_schedule(
    schedule: TemperatureSchedule,
    T0: float,
    Tend: float,
    n_temps: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if n_temps <= 0:
        raise ValueError("`n_temps` must be positive.")

    if callable(schedule):
        temps = torch.tensor([schedule(k) for k in range(n_temps)], dtype=dtype, device=device)
    else:
        name = schedule.lower()
        if n_temps == 1:
            temps = torch.tensor([T0], dtype=dtype, device=device)
        elif name in {"linear", "arithmetic"}:
            temps = torch.linspace(T0, Tend, steps=n_temps, dtype=dtype, device=device)
        elif name in {"geometric", "exp", "exponential"}:
            if T0 <= 0 or Tend <= 0:
                raise ValueError("Geometric schedule requires strictly positive T0 and Tend.")
            base = (Tend / T0) ** (1.0 / (n_temps - 1))
            temps = torch.tensor(
                [T0 * (base ** k) for k in range(n_temps)],
                dtype=dtype,
                device=device,
            )
        else:
            raise ValueError(
                f"Unsupported schedule `{schedule}`. "
                "Choose from 'linear', 'arithmetic', 'geometric', or pass a callable."
            )

    tiny = torch.finfo(dtype).tiny
    return torch.clamp(temps, min=tiny)


@torch.no_grad()
def simulated_annealing(
    env: Env,
    init_solution: Optional[torch.Tensor] = None,
    *,
    batch_size: Optional[int] = None,
    initializer: Optional[Initializer] = None,
    warmup_steps: int = 0,
    n_temps: int = 200,
    steps_per_temp: int = 250,
    schedule: TemperatureSchedule = "geometric",
    T0: float = 1.0,
    Tend: float = 1e-3,
    sweep_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    canonicalize: bool = False,
    return_traces: bool = False,
    neighbor_arg_sampler: Optional[NeighborArgSampler] = None,
) -> Dict[str, Any]:
    """
    Generic simulated annealing loop aligned with the structure used in `SA/`.

    Args:
        env: Environment implementing the `Env` interface.
        init_solution: Optional batch of initial states; if omitted, `initializer`
            must be provided.
        batch_size: Number of parallel chains; required when `init_solution` is None.
        initializer: Callable that builds an initial batch given `(env, batch_size, device, generator)`.
        warmup_steps: Number of outer steps performed at fixed temperature `T0`
            prior to annealing.
        n_temps: Number of temperature levels during annealing.
        steps_per_temp: Outer steps performed at each temperature.
        schedule: Either 'linear'/'arithmetic' or 'geometric', or a callable returning
            temperatures.
        T0: Starting temperature.
        Tend: Final temperature used by built-in schedules.
        sweep_size: Number of Metropolis proposals performed within each inner step.
            Defaults to the environment size `env.N`, yielding a full sweep.
        seed: Optional RNG seed for reproducibility.
        device: Torch device for tensors; defaults to the device of `init_solution`
            or CPU.
        canonicalize: Whether to call `env.canonicalize` after each proposal. The
            canonicalized tensor must preserve shape.
        return_traces: If True, records the running best energy across steps.
        neighbor_arg_sampler: Optional callable returning `(args, kwargs)` to be
            passed to `env.neighbor`. Defaults to `_default_neighbor_args`.

    Returns:
        Dictionary containing the final state, per-chain bests, and (optionally) an
        energy trace mirroring the structure from `SA/`.
    """
    if warmup_steps < 0 or steps_per_temp < 0:
        raise ValueError("`warmup_steps` and `steps_per_temp` must be non-negative.")

    if init_solution is None:
        if batch_size is None or batch_size <= 0:
            raise ValueError("Provide a positive `batch_size` when `init_solution` is None.")
        if initializer is None:
            raise ValueError("`initializer` must be supplied when `init_solution` is None.")
    else:
        if init_solution.ndim < 2:
            raise ValueError("`init_solution` must include a batch dimension.")
        batch_size = init_solution.shape[0]

    if device is None:
        if init_solution is not None:
            device = init_solution.device
        else:
            device = torch.device("cpu")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    if init_solution is None:
        current = initializer(env, batch_size, device, generator)
    else:
        current = init_solution.to(device=device)

    if current.shape[0] != batch_size:
        raise ValueError("Initializer must return a tensor with batch dimension `batch_size`.")

    state_shape = current.shape

    if canonicalize:
        canonical = env.canonicalize(current)
        if canonical.shape != state_shape:
            raise ValueError(
                "Canonicalization must preserve tensor shape. "
                "Disable `canonicalize` or adjust the environment implementation."
            )
        current = canonical

    energy_current = env.energy(current)
    if energy_current.ndim != 1 or energy_current.shape[0] != batch_size:
        raise ValueError("`env.energy` must return a tensor of shape (batch_size,).")

    if energy_current.is_floating_point():
        dtype = energy_current.dtype
    else:
        dtype = torch.float32

    def rand(shape) -> torch.Tensor:
        if generator is None:
            return torch.rand(shape, device=device, dtype=dtype)
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)

    temps = _build_temperature_schedule(schedule, T0, Tend, n_temps, dtype=dtype, device=device)

    if sweep_size is None:
        if not hasattr(env, "N"):
            raise AttributeError(
                "Environment must define attribute `N` or provide `sweep_size` to perform full sweeps."
            )
        sweep_size = getattr(env, "N")
    if not isinstance(sweep_size, int) or sweep_size <= 0:
        raise ValueError("`sweep_size` must be a positive integer.")

    arg_sampler = neighbor_arg_sampler or _default_neighbor_args

    best_state = current.clone()
    best_energy = energy_current.clone()

    best_idx = torch.argmin(best_energy)
    global_best = {
        "state": current[best_idx].clone(),
        "energy": best_energy[best_idx].item(),
    }

    trace = (
        torch.empty(warmup_steps + n_temps * steps_per_temp, device=device, dtype=dtype)
        if return_traces
        else None
    )
    step_counter = 0

    view_shape = (batch_size,) + (1,) * (current.ndim - 1)

    def metropolis_step(temp_tensor: torch.Tensor):
        nonlocal current, energy_current, best_state, best_energy, global_best

        for _ in range(sweep_size):
            extra_args, extra_kwargs = arg_sampler(env, current, generator)
            if not isinstance(extra_args, tuple) or not isinstance(extra_kwargs, dict):
                raise TypeError("`neighbor_arg_sampler` must return (tuple, dict).")

            proposal = env.neighbor(current, *extra_args, **extra_kwargs)
            if proposal.shape != state_shape:
                raise ValueError("`env.neighbor` must return tensors matching the initial state shape.")

            if canonicalize:
                proposal_canonical = env.canonicalize(proposal)
                if proposal_canonical.shape != state_shape:
                    raise ValueError(
                        "Canonicalization must preserve tensor shape. "
                        "Disable `canonicalize` or adjust the environment implementation."
                    )
                proposal = proposal_canonical

            proposal_energy = env.energy(proposal)
            if proposal_energy.shape != energy_current.shape:
                raise ValueError("`env.energy` must return a tensor aligned with the batch dimension.")

            delta = proposal_energy - energy_current
            T_safe = torch.clamp(temp_tensor, min=torch.finfo(dtype).tiny)
            accept = (delta <= 0) | (torch.log(rand((batch_size,))) < (-delta / T_safe))

            if accept.any():
                accept_view = accept.view(view_shape)
                current = torch.where(accept_view, proposal, current)
                energy_current = torch.where(accept, proposal_energy, energy_current)

            improving = energy_current < best_energy
            if improving.any():
                improving_view = improving.view(view_shape)
                best_state = torch.where(improving_view, current, best_state)
                best_energy = torch.where(improving, energy_current, best_energy)

                cand_idx = torch.argmin(best_energy)
                cand_energy = best_energy[cand_idx].item()
                if cand_energy < global_best["energy"]:
                    global_best["state"] = best_state[cand_idx].clone()
                    global_best["energy"] = cand_energy

    T0_tensor = torch.tensor(T0, device=device, dtype=dtype)
    for _ in range(warmup_steps):
        metropolis_step(T0_tensor)
        if trace is not None:
            trace[step_counter] = best_energy.min()
        step_counter += 1

    for temp in temps:
        for _ in range(steps_per_temp):
            metropolis_step(temp)
            if trace is not None:
                trace[step_counter] = best_energy.min()
            step_counter += 1

    result: Dict[str, Any] = {
        "final_state": current,
        "final_energy": energy_current,
        "best_state": best_state,
        "best_energy": best_energy,
        "global_best": global_best,
    }

    if trace is not None:
        result["trace_best"] = trace.detach().cpu()

    return result
