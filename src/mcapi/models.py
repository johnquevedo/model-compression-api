"""Model / tokenizer construction helpers."""

from __future__ import annotations

from typing import Tuple

import torch

from .config import Config


def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path)


def load_teacher(cfg: Config, device: torch.device):
    """Load the (already fine-tuned) teacher for evaluation / soft-label generation."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(cfg.models.teacher)
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer(cfg.models.teacher)
    return model, tokenizer


def load_student(cfg: Config, device: torch.device, from_pretrained: str = None):
    """Load the student. ``from_pretrained`` overrides the base checkpoint (e.g. to resume)."""
    from transformers import AutoModelForSequenceClassification

    source = from_pretrained or cfg.models.student
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=cfg.task.num_labels
    )
    model.to(device)
    tokenizer = load_tokenizer(source)
    return model, tokenizer


def load_classifier(path: str, device: torch.device) -> Tuple[object, object]:
    """Load a saved sequence-classification model + tokenizer from a directory."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
