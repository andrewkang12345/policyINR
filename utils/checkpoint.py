from pathlib import Path
from typing import Any, Dict
import torch


def save_checkpoint(path, model, optimizer=None, extra: Dict[str, Any] | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("extra", {})
