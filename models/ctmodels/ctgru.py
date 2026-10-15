# PyTorch port of raminmh/liquid_time_constant_networks,
# experiments_with_ltcs/ctrnn_model.py::CTGRU (Apache-2.0).
# Modification: decay uses tau, not log(tau); elapsed durations are supported.
"""CT-GRU with the original retrieval/storage gates and timescale bank."""

import math

import torch
from torch import nn

from ._sequence import SequenceModel, cell_timespan, positive_int


class CTGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, *, M=8, cell_clip=-1):
        super().__init__()
        self.input_size = positive_int("input_size", input_size)
        self.hidden_size = positive_int("hidden_size", hidden_size)
        self.M = positive_int("M", M)
        self.state_size = hidden_size * M
        self.output_size = hidden_size
        self.cell_clip = cell_clip
        self.register_buffer("ln_tau_table", torch.arange(M, dtype=torch.float32) * (0.5 * math.log(10.0)))
        self.tau_r = nn.Linear(input_size + hidden_size, hidden_size * M)
        self.tau_s = nn.Linear(input_size + hidden_size, hidden_size * M)
        self.detect_signal = nn.Linear(input_size + hidden_size, hidden_size)
        for layer in (self.tau_r, self.tau_s, self.detect_signal):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs, state, elapsed_time=1.0):
        h_hat = state.reshape(-1, self.hidden_size, self.M)
        h = h_hat.sum(dim=-1)
        fused_input = torch.cat([inputs, h], dim=-1)
        ln_tau_r = self.tau_r(fused_input).reshape(-1, self.hidden_size, self.M)
        rki = torch.softmax(-(ln_tau_r - self.ln_tau_table).square(), dim=-1)
        q_input = (rki * h_hat).sum(dim=-1)
        qk = torch.tanh(self.detect_signal(torch.cat([inputs, q_input], dim=-1)))
        ln_tau_s = self.tau_s(fused_input).reshape(-1, self.hidden_size, self.M)
        ski = torch.softmax(-(ln_tau_s - self.ln_tau_table).square(), dim=-1)
        duration = cell_timespan(elapsed_time, inputs)
        if duration.ndim > 0:
            duration = duration.unsqueeze(-1)
        # Upstream incorrectly divides by ln_tau_table (whose first entry is 0).
        decay = torch.exp(-duration / self.ln_tau_table.exp())
        h_hat_next = ((1.0 - ski) * h_hat + ski * qk.unsqueeze(-1)) * decay
        if self.cell_clip > 0:
            h_hat_next = h_hat_next.clamp(-self.cell_clip, self.cell_clip)
        return h_hat_next.sum(dim=-1), h_hat_next.flatten(start_dim=1)


class CTGRU(SequenceModel):
    """CT-GRU with flattened (B,H*M) state and (B,H) cell output."""

    def __init__(
        self, input_size, hidden_size, output_size=None, *,
        batch_first=True, return_sequences=True, M=8, cell_clip=-1,
    ):
        cell = CTGRUCell(input_size, hidden_size, M=M, cell_clip=cell_clip)
        super().__init__(
            input_size, hidden_size, cell, output_size=output_size,
            batch_first=batch_first, return_sequences=return_sequences,
            state_size=hidden_size * M,
        )
