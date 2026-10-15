# PyTorch port of raminmh/liquid_time_constant_networks,
# experiments_with_ltcs/ctrnn_model.py (Apache-2.0).
# Modification: learned tau is used in the ODE; elapsed durations are supported.
"""Continuous-time RNN using the original explicit Euler solver."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._sequence import SequenceModel, cell_timespan, positive_int


class CTRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, *, cell_clip=-1,
                 global_feedback=False, fix_tau=True, tau=1.0,
                 ode_unfolds=6, delta_t=0.1):
        super().__init__()
        self.input_size = positive_int("input_size", input_size)
        self.hidden_size = positive_int("hidden_size", hidden_size)
        self.ode_unfolds = positive_int("ode_unfolds", ode_unfolds)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        if not math.isfinite(delta_t) or delta_t <= 0:
            raise ValueError("delta_t must be finite and positive")
        self.delta_t = delta_t
        self.cell_clip = cell_clip
        self.global_feedback = global_feedback
        self.fix_tau = fix_tau
        features = input_size + hidden_size if global_feedback else input_size
        self.linear = nn.Linear(features, hidden_size)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if fix_tau:
            self.register_buffer("tau", torch.tensor(float(tau)))
        else:
            # Preserve the TF raw-parameter initialization: effective tau=softplus(tau).
            self.raw_tau = nn.Parameter(torch.tensor(float(tau)))

    def forward(self, inputs, state, elapsed_time=None):
        elapsed_time = self.ode_unfolds * self.delta_t if elapsed_time is None else elapsed_time
        step = cell_timespan(elapsed_time, inputs) / self.ode_unfolds
        tau = self.tau if self.fix_tau else F.softplus(self.raw_tau)
        if not self.global_feedback:
            drive = torch.tanh(self.linear(inputs))
        for _ in range(self.ode_unfolds):
            if self.global_feedback:
                drive = torch.tanh(self.linear(torch.cat([inputs, state], dim=-1)))
            state = state + step * (-state / tau + drive)
            if self.cell_clip > 0:
                state = state.clamp(-self.cell_clip, self.cell_clip)
        return state, state


class CTRNN(SequenceModel):
    """Euler CTRNN; default duration is 6*0.1=0.6 as in the original code."""

    def __init__(
        self, input_size, hidden_size, output_size=None, *,
        batch_first=True, return_sequences=True, cell_clip=-1,
        global_feedback=False, fix_tau=True, tau=1.0,
        ode_unfolds=6, delta_t=0.1,
    ):
        cell = CTRNNCell(
            input_size, hidden_size, cell_clip=cell_clip,
            global_feedback=global_feedback, fix_tau=fix_tau, tau=tau,
            ode_unfolds=ode_unfolds, delta_t=delta_t,
        )
        super().__init__(
            input_size, hidden_size, cell, output_size=output_size,
            batch_first=batch_first, return_sequences=return_sequences,
            default_timespan=ode_unfolds * delta_t,
        )
