from __future__ import annotations

import torch


def poisson_encode(x: torch.Tensor, timesteps: int, dt: float = 1.0) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError("timesteps must be > 0")

    x = torch.clamp(x, 0.0, 1.0)
    p = x * dt

    r = torch.rand((timesteps,) + tuple(x.shape), device=x.device, dtype=x.dtype)
    return (r < p).to(x.dtype)


def ttfs_encode(x: torch.Tensor, timesteps: int, eps: float = 1e-6) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError("timesteps must be > 0")

    x = torch.clamp(x, 0.0, 1.0)
    t = (1.0 - x) * (timesteps - 1)
    t = torch.round(t).to(torch.int64)

    spikes = torch.zeros((timesteps,) + tuple(x.shape), device=x.device, dtype=x.dtype)

    flat_spikes = spikes.view(timesteps, -1)
    flat_t = t.view(-1)

    active = x.view(-1) > eps
    idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return spikes

    times = flat_t[idx]
    flat_spikes[times, idx] = 1.0

    return spikes
