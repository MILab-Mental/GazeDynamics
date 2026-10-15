# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from ncps/ncps/torch/ltc_cell.py for direct dense initialization.
# Connection objects and sparsity masks removed; LTC dynamics retained.
import torch
from torch import nn
import numpy as np

from ._sequence import SequenceModel, positive_int


class LTCCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=False,
        erev_init_seed=1111,
    ):
        """A `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ cell.

        .. Note::
            This cell processes a single time step. Use `ctmodels.LTC` for sequences.


        :param input_size: Number of input features.
        :param hidden_size: Number of fully connected hidden units.
        :param input_mapping:
        :param output_mapping:
        :param ode_unfolds:
        :param epsilon:
        :param implicit_param_constraints:
        :param erev_init_seed: NumPy seed for reversal-potential initialization.
        """
        super(LTCCell, self).__init__()
        self.input_size = positive_int("input_size", input_size)
        self.hidden_size = positive_int("hidden_size", hidden_size)
        positive_int("ode_unfolds", ode_unfolds)
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if input_mapping not in (None, "none", "linear", "affine"):
            raise ValueError("input_mapping must be None, 'none', 'linear', or 'affine'")
        if output_mapping not in (None, "none", "linear", "affine"):
            raise ValueError("output_mapping must be None, 'none', 'linear', or 'affine'")
        self.make_positive_fn = (
            nn.Softplus() if implicit_param_constraints else nn.Identity()
        )
        self._implicit_param_constraints = implicit_param_constraints
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._erev_init_seed = erev_init_seed
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = torch.nn.ReLU()
        self._allocate_parameters()

    @property
    def state_size(self):
        return self.hidden_size

    @property
    def sensory_size(self):
        return self.input_size

    @property
    def motor_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.motor_size

    def add_weight(self, name, init_value, requires_grad=True):
        param = torch.nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        rng = np.random.default_rng(self._erev_init_seed)
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(rng.choice([-1, 1, 1], size=(self.state_size, self.state_size))),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(rng.choice([-1, 1, 1], size=(self.sensory_size, self.state_size))),
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                init_value=torch.ones((self.sensory_size,)),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                init_value=torch.zeros((self.sensory_size,)),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                init_value=torch.ones((self.motor_size,)),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                init_value=torch.zeros((self.motor_size,)),
            )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant
        cm_t = self.make_positive_fn(self._params["cm"]) / (
            elapsed_time / self._ode_unfolds
        )

        # Unfold the multiply ODE multiple times into one RNN step
        w_param = self.make_positive_fn(self._params["w"])
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            gleak = self.make_positive_fn(self._params["gleak"])
            numerator = cm_t * v_pre + gleak * self._params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def apply_weight_constraints(self):
        if not self._implicit_param_constraints:
            # In implicit mode, the parameter constraints are implemented via
            # a softplus function at runtime
            self._params["w"].data = self._clip(self._params["w"].data)
            self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
            self._params["cm"].data = self._clip(self._params["cm"].data)
            self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, inputs, states, elapsed_time=1.0):
        # Regularly sampled mode (elapsed time = 1 second)
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states, elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, next_state


class LTC(SequenceModel):
    """Fully connected LTC with direct dimension-based parameter initialization.

    Implicit positivity constraints match the ncps LTC sequence defaults.
    With implicit_param_constraints=False, call apply_weight_constraints()
    after each optimizer step.
    """

    def __init__(
        self, input_size, hidden_size, output_size=None, *,
        batch_first=True, return_sequences=True, mixed_memory=False,
        input_mapping="affine", output_mapping="affine", ode_unfolds=6,
        epsilon=1e-8, implicit_param_constraints=True,
        erev_init_seed=1111,
    ):
        cell = LTCCell(
            input_size, hidden_size, input_mapping=input_mapping,
            output_mapping=output_mapping, ode_unfolds=ode_unfolds,
            epsilon=epsilon, implicit_param_constraints=implicit_param_constraints,
            erev_init_seed=erev_init_seed,
        )
        super().__init__(
            input_size, hidden_size, cell, output_size=output_size,
            batch_first=batch_first, return_sequences=return_sequences,
            mixed_memory=mixed_memory,
        )

    def apply_weight_constraints(self):
        self.rnn_cell.apply_weight_constraints()
