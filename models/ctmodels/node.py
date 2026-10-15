# PyTorch port of raminmh/liquid_time_constant_networks,
# experiments_with_ltcs/ctrnn_model.py::NODE (Apache-2.0).
# Modification: explicit elapsed durations and a common sequence interface.
"""Original recurrent Neural ODE baseline with a fixed-step RK4 solver."""

import math

import torch
from torch import nn

from ._sequence import SequenceModel, cell_timespan, positive_int


class NODECell(nn.Module):
    def __init__(self, input_size, hidden_size, *, cell_clip=-1,
                 ode_unfolds=6, delta_t=0.1):
        super().__init__()
        self.input_size = positive_int("input_size", input_size)
        self.hidden_size = positive_int("hidden_size", hidden_size)
        self.ode_unfolds = positive_int("ode_unfolds", ode_unfolds)
        if not math.isfinite(delta_t) or delta_t <= 0:
            raise ValueError("delta_t must be finite and positive")
        self.delta_t = delta_t
        self.cell_clip = cell_clip
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def _f_prime(self, inputs, state):
        return torch.tanh(self.linear(torch.cat([inputs, state], dim=-1)))

    def forward(self, inputs, state, elapsed_time=None):
        elapsed_time = self.ode_unfolds * self.delta_t if elapsed_time is None else elapsed_time
        step = cell_timespan(elapsed_time, inputs) / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            k1 = step * self._f_prime(inputs, state)
            k2 = step * self._f_prime(inputs, state + k1 * 0.5)
            k3 = step * self._f_prime(inputs, state + k2 * 0.5)
            k4 = step * self._f_prime(inputs, state + k3)
            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if self.cell_clip > 0:
                state = state.clamp(-self.cell_clip, self.cell_clip)
        return state, state


class NODE(SequenceModel):
    """RK4 Neural ODE RNN; default duration matches the upstream 0.6."""

    def __init__(
        self, input_size, hidden_size, output_size=None, *,
        batch_first=True, return_sequences=True, cell_clip=-1,
        ode_unfolds=6, delta_t=0.1,
    ):
        cell = NODECell(
            input_size, hidden_size, cell_clip=cell_clip,
            ode_unfolds=ode_unfolds, delta_t=delta_t,
        )
        super().__init__(
            input_size, hidden_size, cell, output_size=output_size,
            batch_first=batch_first, return_sequences=return_sequences,
            default_timespan=ode_unfolds * delta_t,
        )
