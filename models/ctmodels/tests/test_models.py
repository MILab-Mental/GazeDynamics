"""Behavioral and numerical checks for the common API and upstream ports."""

import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from ctmodels import MODEL_REGISTRY, CTGRUCell, CTRNNCell, NODECell, build_model

torch.set_num_threads(1)
NAMES = tuple(MODEL_REGISTRY)
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def state_tensors(state):
    return state if isinstance(state, tuple) else (state,)


def assert_state_close(left, right):
    assert len(state_tensors(left)) == len(state_tensors(right))
    for a, b in zip(state_tensors(left), state_tensors(right)):
        torch.testing.assert_close(a, b)


def load_original(relative_path):
    path = SOURCE_ROOT / relative_path
    if not path.exists():
        pytest.skip("Original source checkout is not present")
    spec = importlib.util.spec_from_file_location("reference_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("batch", [1, 2, 5])
def test_forward_backward_and_independent_batch(name, batch):
    torch.manual_seed(7)
    model = build_model(name, 3, 5, output_size=2)
    x = torch.randn(batch, 4, 3, requires_grad=True)
    dt = torch.linspace(0.1, 0.9, batch * 4).reshape(batch, 4)
    output, state = model(x, timespans=dt)
    assert output.shape == (batch, 4, 2)
    expected_size = 40 if name == "ctgru" else 5
    for tensor in state_tensors(state):
        assert tensor.shape == (batch, expected_size)
        assert torch.isfinite(tensor).all()
    # Different per-sample durations must give the same result when run separately.
    for i in range(batch):
        single, single_state = model(x[i:i + 1], timespans=dt[i:i + 1])
        torch.testing.assert_close(single, output[i:i + 1])
        for actual, full in zip(state_tensors(single_state), state_tensors(state)):
            torch.testing.assert_close(actual, full[i:i + 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    optimizer.step()
    updated, _ = model(x.detach(), timespans=dt)
    assert torch.isfinite(updated).all()


@pytest.mark.parametrize("name", NAMES)
def test_layout_last_output_chunking_and_serialization(name):
    torch.manual_seed(8)
    model = build_model(name, 3, 5, output_size=2).double()
    x = torch.randn(2, 5, 3, dtype=torch.float64)
    dt = torch.rand(2, 5, dtype=torch.float64) + 0.1
    full, final = model(x, timespans=dt)
    assert full.dtype == torch.float64
    head, state = model(x[:, :2], timespans=dt[:, :2])
    tail, state = model(x[:, 2:], hx=state, timespans=dt[:, 2:])
    torch.testing.assert_close(torch.cat([head, tail], dim=1), full)
    assert_state_close(state, final)
    for first in (True, False):
        model.batch_first = first
        bx = x if first else x.transpose(0, 1)
        bdt = dt if first else dt.transpose(0, 1)
        for last_only in (False, True):
            model.return_sequences = not last_only
            out, state = model(bx, timespans=bdt.unsqueeze(-1))
            expected = full[:, -1] if last_only else (full if first else full.transpose(0, 1))
            torch.testing.assert_close(out, expected)
            assert_state_close(state, final)
            # Unbatched last-output mode must keep the feature dimension, including O=1.
            single, single_state = model(x[0], timespans=dt[0].unsqueeze(-1))
            torch.testing.assert_close(single, full[0, -1] if last_only else full[0])
            for s, full_s in zip(state_tensors(single_state), state_tensors(final)):
                torch.testing.assert_close(s, full_s[0])
            zero = tuple(torch.zeros_like(s) for s in state_tensors(single_state))
            hx = zero if isinstance(single_state, tuple) else zero[0]
            again, _ = model(x[0], hx=hx, timespans=dt[0])
            torch.testing.assert_close(again, single)
    model.batch_first, model.return_sequences = True, True
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build_model(name, 3, 5, output_size=2).double()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    result, state = restored(x, timespans=dt)
    torch.testing.assert_close(result, full)
    assert_state_close(state, final)


@pytest.mark.parametrize("name", NAMES)
def test_duration_forms_and_effect(name):
    model = build_model(name, 3, 5)
    x = torch.randn(2, 4, 3)
    expected, state = model(x, timespans=0.2)
    for dt in (torch.full((4,), 0.2), torch.full((2, 4), 0.2), torch.full((2, 4, 1), 0.2)):
        actual, actual_state = model(x, timespans=dt)
        torch.testing.assert_close(actual, expected)
        assert_state_close(actual_state, state)
    changed, _ = model(x, timespans=1.0)
    if name == "lstm":
        torch.testing.assert_close(changed, expected)
    else:
        assert not torch.allclose(changed, expected)
    default, _ = model(x)
    explicit, _ = model(x, timespans=model.default_timespan)
    torch.testing.assert_close(default, explicit)


@pytest.mark.parametrize("mode", ["default", "pure", "no_gate"])
@pytest.mark.parametrize("mixed", [False, True])
def test_cfc_original_sequence_equivalence(mode, mixed):
    original = load_original("ncps/ncps/torch/cfc_cell.py")
    orig_lstm = load_original("ncps/ncps/torch/lstm.py")
    model = build_model("cfc", 3, 5, mode=mode, mixed_memory=mixed).double()
    ref = original.CfCCell(3, 5, mode=mode).double()
    ref.load_state_dict(model.rnn_cell.state_dict())
    if mixed:
        memory = orig_lstm.LSTMCell(3, 5).double()
        memory.load_state_dict(model.lstm.state_dict())
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    dt = torch.rand(2, 4, dtype=torch.float64) + 0.1
    h, c = x.new_zeros(2, 5), x.new_zeros(2, 5)
    outputs = []
    for t in range(4):
        if mixed:
            h, c = memory(x[:, t], (h, c))
        out, h = ref(x[:, t], h, dt[:, t:t + 1])
        outputs.append(out)
    result, state = model(x, timespans=dt)
    torch.testing.assert_close(result, torch.stack(outputs, dim=1), rtol=0, atol=0)
    assert_state_close(state, (h, c) if mixed else h)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("seed", [1111, 27])
def test_ltc_original_sequence_equivalence(mixed, implicit, seed):
    original = load_original("ncps/ncps/torch/ltc_cell.py")
    wiring = load_original("ncps/ncps/wirings/wirings.py")
    orig_lstm = load_original("ncps/ncps/torch/lstm.py")
    model = build_model(
        "ltc", 3, 5, mixed_memory=mixed, implicit_param_constraints=implicit,
        erev_init_seed=seed,
    ).double()
    ref = original.LTCCell(
        wiring.FullyConnected(5, erev_init_seed=seed),
        in_features=3, implicit_param_constraints=implicit,
    ).double()
    # Dense initialization must preserve the upstream reversal-potential signs.
    torch.testing.assert_close(model.rnn_cell.erev, ref.erev, rtol=0, atol=0)
    torch.testing.assert_close(model.rnn_cell.sensory_erev, ref.sensory_erev, rtol=0, atol=0)
    loaded = ref.load_state_dict(model.rnn_cell.state_dict(), strict=False)
    assert set(loaded.missing_keys) == {"sparsity_mask", "sensory_sparsity_mask"}
    assert loaded.unexpected_keys == []
    if mixed:
        memory = orig_lstm.LSTMCell(3, 5).double()
        memory.load_state_dict(model.lstm.state_dict())
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    dt = torch.rand(2, 4, dtype=torch.float64) + 0.1
    h, c = x.new_zeros(2, 5), x.new_zeros(2, 5)
    outputs = []
    for t in range(4):
        if mixed:
            h, c = memory(x[:, t], (h, c))
        out, h = ref(x[:, t], h, dt[:, t:t + 1])
        outputs.append(out)
    result, state = model(x, timespans=dt)
    torch.testing.assert_close(result, torch.stack(outputs, dim=1), rtol=0, atol=0)
    assert_state_close(state, (h, c) if mixed else h)
    model.apply_weight_constraints()


def test_lstm_original_sequence_equivalence():
    original = load_original("ncps/ncps/torch/lstm.py")
    model = build_model("lstm", 3, 5).double()
    ref = original.LSTMCell(3, 5).double()
    ref.load_state_dict(model.rnn_cell.state_dict())
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    state = (x.new_zeros(2, 5), x.new_zeros(2, 5))
    outputs = []
    for t in range(4):
        state = ref(x[:, t], state)
        outputs.append(state[0])
    result, actual_state = model(x)
    torch.testing.assert_close(result, torch.stack(outputs, dim=1), rtol=0, atol=0)
    assert_state_close(actual_state, state)


@pytest.mark.parametrize("name", ["ltc", "cfc"])
@pytest.mark.parametrize("batch_first", [False, True])
def test_mixed_memory_state_continuation(name, batch_first):
    model = build_model(name, 3, 5, mixed_memory=True, batch_first=batch_first).double()
    x = torch.randn(4, 3, dtype=torch.float64)
    dt = torch.rand(4, dtype=torch.float64) + 0.1
    full, final = model(x, timespans=dt)
    head, state = model(x[:2], timespans=dt[:2])
    tail, state = model(x[2:], hx=state, timespans=dt[2:])
    torch.testing.assert_close(torch.cat([head, tail], dim=0), full)
    assert_state_close(state, final)
    model.return_sequences = False
    last, state = model(x, timespans=dt)
    torch.testing.assert_close(last, full[-1])
    assert_state_close(state, final)


def np_linear(layer, x):
    return x @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()


@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_ctrnn_numpy_euler_and_learned_tau(feedback, fixed):
    cell = CTRNNCell(3, 5, global_feedback=feedback, fix_tau=fixed).double()
    x, h = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
    dt = torch.tensor([0.2, 0.8], dtype=torch.float64)
    xn, expected = x.numpy(), h.numpy().copy()
    tau = 1.0 if fixed else np.logaddexp(0.0, 1.0)
    for _ in range(6):
        fused = np.concatenate([xn, expected], axis=-1) if feedback else xn
        drive = np.tanh(np_linear(cell.linear, fused))
        expected += dt.numpy()[:, None] / 6 * (-expected / tau + drive)
    output, state = cell(x, h, dt)
    np.testing.assert_allclose(output.detach().numpy(), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(output, state)
    if not fixed:
        output.square().sum().backward()
        assert cell.raw_tau.grad is not None and cell.raw_tau.grad.abs() > 0


def test_node_numpy_rk4():
    cell = NODECell(3, 5).double()
    x, h = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
    dt = np.array([[0.2], [0.8]])
    expected = h.numpy().copy()
    def derivative(v):
        return np.tanh(np_linear(cell.linear, np.concatenate([x.numpy(), v], axis=-1)))
    for _ in range(6):
        k1 = dt / 6 * derivative(expected)
        k2 = dt / 6 * derivative(expected + 0.5 * k1)
        k3 = dt / 6 * derivative(expected + 0.5 * k2)
        k4 = dt / 6 * derivative(expected + k3)
        expected += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    output, _ = cell(x, h, torch.from_numpy(dt))
    np.testing.assert_allclose(output.detach().numpy(), expected, rtol=1e-12, atol=1e-12)


def test_ctgru_numpy_timescale_gates_and_decay():
    cell = CTGRUCell(3, 2, M=3).double()
    x, state = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 6, dtype=torch.float64)
    dt = torch.tensor([0.3, 0.9], dtype=torch.float64)
    h_hat = state.numpy().reshape(2, 2, 3)
    fused = np.concatenate([x.numpy(), h_hat.sum(axis=-1)], axis=-1)
    table = cell.ln_tau_table.numpy()
    def gate(layer):
        score = -(np_linear(layer, fused).reshape(2, 2, 3) - table) ** 2
        score = np.exp(score - score.max(axis=-1, keepdims=True))
        return score / score.sum(axis=-1, keepdims=True)
    retrieved = (gate(cell.tau_r) * h_hat).sum(axis=-1)
    q = np.tanh(np_linear(cell.detect_signal, np.concatenate([x.numpy(), retrieved], axis=-1)))
    stored = gate(cell.tau_s)
    expected = ((1 - stored) * h_hat + stored * q[..., None]) * np.exp(-dt.numpy()[:, None, None] / np.exp(table))
    output, new_state = cell(x, state, dt)
    np.testing.assert_allclose(new_state.detach().numpy().reshape(2, 2, 3), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(output.detach().numpy(), expected.sum(axis=-1), rtol=1e-12, atol=1e-12)
    assert (new_state.reshape(2, 2, 3)[:, :, 0].abs() > 0).all()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad_dt", [0.0, -1.0, float("nan"), float("inf"), torch.ones(2, 3)])
def test_invalid_durations(name, bad_dt):
    with pytest.raises(ValueError, match="timespans"):
        build_model(name, 3, 5)(torch.randn(2, 4, 3), timespans=bad_dt)


@pytest.mark.parametrize("name", NAMES)
def test_invalid_states_and_single_feature_last_output(name):
    model = build_model(name, 3, 5, output_size=1, return_sequences=False)
    with pytest.raises(ValueError, match="hx|state"):
        model(torch.randn(2, 4, 3), hx=torch.zeros(2, 6))
    output, _ = model(torch.randn(4, 3))
    assert output.shape == (1,)
    with pytest.raises(ValueError, match="time step"):
        model(torch.randn(2, 0, 3))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_backward_matches_cpu(name):
    cpu = build_model(name, 3, 5, output_size=2)
    gpu = build_model(name, 3, 5, output_size=2).cuda()
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(2, 4, 3)
    dt = torch.rand(2, 4) + 0.1
    expected, expected_state = cpu(x, timespans=dt)
    gx = x.cuda().requires_grad_()
    output, state = gpu(gx, timespans=dt)
    torch.testing.assert_close(output.cpu(), expected, atol=1e-5, rtol=1e-4)
    for actual, reference in zip(state_tensors(state), state_tensors(expected_state)):
        torch.testing.assert_close(actual.cpu(), reference, atol=1e-5, rtol=1e-4)
    output.square().mean().backward()
    assert torch.isfinite(gx.grad).all()
    for p in gpu.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_factory_validation():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("missing", 3, 5)
    with pytest.raises(ValueError, match="positive integer"):
        build_model("cfc", 3, 0)
    with pytest.raises(TypeError):
        build_model("node", 3, 5, unsupported=True)
    assert isinstance(build_model(" CfC ", 3, 5), MODEL_REGISTRY["cfc"])
