"""Shared helpers: device selection, seeding, sizing, and JSON I/O."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available device: explicit override > cuda > mps > cpu."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def dir_size_mb(path: str) -> float:
    """Total size (MB) of all files under ``path`` — used for on-disk model size."""
    if not os.path.exists(path):
        return 0.0
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)
