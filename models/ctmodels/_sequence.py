"""Shared sequence adapter; model equations live in the individual cells."""

import torch
from torch import nn


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def cell_timespan(value, inputs):
    """Normalize a scalar or per-sample duration to (batch, 1)."""
    value = torch.as_tensor(value, dtype=inputs.dtype, device=inputs.device)
    if value.ndim == 0:
        return value
    if value.shape == (inputs.shape[0],):
        return value.unsqueeze(-1)
    if value.shape == (inputs.shape[0], 1):
        return value
    raise ValueError("Cell duration must be a scalar, (batch,), or (batch, 1)")


class SequenceModel(nn.Module):
    """Single-layer RNN with a common (output, final_state) interface.

    Inputs are (B,T,F), (T,B,F) when batch_first=False, or unbatched
    (T,F). timespans are elapsed durations, never absolute timestamps.
    """

    def __init__(
        self, input_size, hidden_size, cell, *, output_size=None,
        batch_first=True, return_sequences=True, state_size=None,
        mixed_memory=False, lstm=False, default_timespan=1.0,
    ):
        super().__init__()
        self.input_size = positive_int("input_size", input_size)
        self.hidden_size = positive_int("hidden_size", hidden_size)
        self.state_size = state_size if state_size is not None else hidden_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.use_mixed = mixed_memory
        self._is_lstm = lstm
        self.default_timespan = default_timespan
        self.rnn_cell = cell
        cell_output_size = getattr(cell, "output_size", hidden_size)
        self.output_size = (
            cell_output_size if output_size is None
            else positive_int("output_size", output_size)
        )
        self.fc = (
            nn.Identity() if output_size is None
            else nn.Linear(cell_output_size, self.output_size)
        )
        if mixed_memory:
            from .lstm import LSTMCell
            self.lstm = LSTMCell(input_size, hidden_size)

    def _timespans(self, timespans, inputs, unbatched):
        batch, length = inputs.shape[:2]
        if timespans is None:
            timespans = self.default_timespan
        ts = torch.as_tensor(timespans, dtype=inputs.dtype, device=inputs.device)
        if ts.ndim == 0:
            ts = ts.expand(batch, length)
        elif ts.ndim == 1 and ts.shape == (length,):
            ts = ts.unsqueeze(0).expand(batch, length)
        else:
            if unbatched:
                if ts.shape != (length, 1):
                    raise ValueError("Unbatched timespans must have shape (T,) or (T,1)")
                ts = ts.reshape(1, length)
            else:
                if ts.ndim == 3 and ts.shape[-1] == 1:
                    ts = ts.squeeze(-1)
                if not self.batch_first and ts.ndim == 2:
                    ts = ts.transpose(0, 1)
                if ts.shape != (batch, length):
                    layout = "(B,T)" if self.batch_first else "(T,B)"
                    raise ValueError(f"timespans must have shape {layout} or {layout[:-1]},1)")
        if not torch.isfinite(ts).all() or (ts <= 0).any():
            raise ValueError("timespans must contain finite, strictly positive durations")
        return ts

    def _state(self, hx, inputs, unbatched):
        batch = inputs.shape[0]
        tuple_state = self.use_mixed or self._is_lstm
        if hx is None:
            h = inputs.new_zeros(batch, self.state_size)
            return (h, inputs.new_zeros(batch, self.hidden_size)) if tuple_state else h
        if tuple_state:
            if not isinstance(hx, (tuple, list)) or len(hx) != 2:
                raise ValueError("This model requires hx=(h0,c0)")
            states = tuple(hx)
            sizes = (self.state_size, self.hidden_size)
        else:
            states, sizes = (hx,), (self.state_size,)
        normalized = []
        for state, size in zip(states, sizes):
            expected = (size,) if unbatched else (batch, size)
            if not isinstance(state, torch.Tensor) or tuple(state.shape) != expected:
                raise ValueError(f"Each state tensor must have shape {expected}")
            if state.device != inputs.device or state.dtype != inputs.dtype:
                raise ValueError("hx must have the same device and dtype as the input")
            normalized.append(state.unsqueeze(0) if unbatched else state)
        return tuple(normalized) if tuple_state else normalized[0]

    def forward(self, input, hx=None, timespans=None):
        if input.ndim not in (2, 3) or input.shape[-1] != self.input_size:
            raise ValueError(f"input must have shape (T,{self.input_size}) or a 3-D sequence")
        if not input.is_floating_point():
            raise ValueError("input must be a floating point tensor")
        unbatched = input.ndim == 2
        inputs = input.unsqueeze(0) if unbatched else input
        if not unbatched and not self.batch_first:
            inputs = inputs.transpose(0, 1)
        if inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise ValueError("input must contain at least one sample and one time step")
        ts = self._timespans(timespans, inputs, unbatched)
        state = self._state(hx, inputs, unbatched)
        outputs = []
        for t in range(inputs.shape[1]):
            x = inputs[:, t]
            if self._is_lstm:
                state = self.rnn_cell(x, state)
                out = state[0]
            elif self.use_mixed:
                h, c = self.lstm(x, state)
                out, h = self.rnn_cell(x, h, ts[:, t:t + 1])
                state = (h, c)
            else:
                out, state = self.rnn_cell(x, state, ts[:, t:t + 1])
            out = self.fc(out)
            if self.return_sequences:
                outputs.append(out)
        output = torch.stack(outputs, dim=1) if self.return_sequences else out
        if unbatched:
            output = output[0]
            state = tuple(s[0] for s in state) if isinstance(state, tuple) else state[0]
        elif self.return_sequences and not self.batch_first:
            output = output.transpose(0, 1)
        return output, state
