"""Factory for the six PyTorch sequence models."""

from .lstm import LSTM
from .ctrnn import CTRNN
from .ltc import LTC
from .cfc import CfC
from .ctgru import CTGRU
from .node import NODE


MODEL_REGISTRY = {
    "lstm": LSTM,
    "ctrnn": CTRNN,
    "ltc": LTC,
    "cfc": CfC,
    "ctgru": CTGRU,
    "node": NODE,
}


def build_model(model_name, input_size, hidden_size, output_size=None, **kwargs):
    """Build a model by name, forwarding model-specific keyword arguments.

    Example: build_model('ltc', 32, 64, output_size=3, ode_unfolds=6).
    Every model returns (output, final_state) from forward(input,hx,timespans).
    """
    if not isinstance(model_name, str):
        raise ValueError("model_name must be a string")
    name = model_name.strip().lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}; choose from {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](input_size, hidden_size, output_size=output_size, **kwargs)
