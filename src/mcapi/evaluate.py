"""Evaluate any sequence-classification model on the eval DataLoader."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics


@torch.no_grad()
def evaluate_model(model, eval_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Run the model over the eval set and return accuracy / macro-F1."""
    model.eval()
    all_preds = []
    all_labels = []
    for batch in eval_loader:
        batch = batch.to(device)
        logits = model(**batch.model_inputs()).logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.labels.cpu().tolist())
    return compute_metrics(all_preds, all_labels)
