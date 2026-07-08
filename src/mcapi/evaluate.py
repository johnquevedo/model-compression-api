"""Evaluate any sequence-classification model on the eval DataLoader."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics


def model_inputs_for(model, batch) -> Dict[str, torch.Tensor]:
    """Return only the batch inputs accepted by the model family."""
    inputs = batch.model_inputs()
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if "distilbert" in model_type.lower():
        inputs.pop("token_type_ids", None)
    return inputs


@torch.no_grad()
def evaluate_model(model, eval_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Run the model over the eval set and return accuracy / macro-F1."""
    model.eval()
    all_preds = []
    all_labels = []
    for batch in eval_loader:
        batch = batch.to(device)
        logits = model(**model_inputs_for(model, batch)).logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.labels.cpu().tolist())
    return compute_metrics(all_preds, all_labels)
