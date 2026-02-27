from __future__ import annotations

from dataclasses import dataclass

import torch

from .surrogate import SurrogateSpike


@dataclass
class LIFState:
    v: torch.Tensor
    refractory: torch.Tensor


class LIFLayer(torch.nn.Module):
    def __init__(
        self,
        size: int,
        tau: float = 0.95,
        threshold: float = 1.0,
        reset: float = 0.0,
        refractory_steps: int = 0,
        surrogate_beta: float = 10.0,
    ):
        super().__init__()
        self.size = int(size)
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.reset = float(reset)
        self.refractory_steps = int(refractory_steps)
        self.spike_fn = SurrogateSpike(threshold=self.threshold, beta=surrogate_beta)

    def init_state(self, batch_size: int, device=None, dtype=None) -> LIFState:
        v = torch.zeros(batch_size, self.size, device=device, dtype=dtype)
        refractory = torch.zeros(batch_size, self.size, device=device, dtype=torch.int64)
        return LIFState(v=v, refractory=refractory)

    def forward(self, input_current: torch.Tensor, state: LIFState) -> tuple[torch.Tensor, LIFState]:
        if input_current.ndim != 2 or input_current.shape[-1] != self.size:
            raise ValueError(f"Expected input_current shape [B, {self.size}], got {tuple(input_current.shape)}")

        not_refractory = (state.refractory <= 0).to(input_current.dtype)

        v = self.tau * state.v + input_current
        v = v * not_refractory + state.v * (1.0 - not_refractory)

        spikes = self.spike_fn(v)

        if self.refractory_steps > 0:
            new_refractory = torch.clamp(state.refractory - 1, min=0)
            new_refractory = torch.where(
                spikes.to(torch.bool),
                torch.full_like(new_refractory, self.refractory_steps),
                new_refractory,
            )
        else:
            new_refractory = torch.zeros_like(state.refractory)

        v = torch.where(spikes.to(torch.bool), torch.full_like(v, self.reset), v)

        return spikes, LIFState(v=v, refractory=new_refractory)
