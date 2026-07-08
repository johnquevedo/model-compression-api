"""Dataset loading and preprocessing.

Wraps a Hugging Face text-classification dataset (default: GLUE/SST-2) and turns it
into tokenized PyTorch DataLoaders that both the fine-tuning and distillation loops use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor]
    labels: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_type_ids=None if self.token_type_ids is None else self.token_type_ids.to(device),
            labels=self.labels.to(device),
        )

    def model_inputs(self) -> Dict[str, torch.Tensor]:
        inputs = {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
        if self.token_type_ids is not None:
            inputs["token_type_ids"] = self.token_type_ids
        return inputs


class TokenizedDataset(Dataset):
    """Holds pre-tokenized tensors in memory (SST-2 is tiny, so this is fine)."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    keys = batch[0].keys()
    stacked = {k: torch.stack([b[k] for b in batch]) for k in keys}
    return Batch(
        input_ids=stacked["input_ids"],
        attention_mask=stacked["attention_mask"],
        token_type_ids=stacked.get("token_type_ids"),
        labels=stacked["labels"],
    )


def _tokenize_split(dataset, tokenizer, cfg: Config, limit: Optional[int]) -> TokenizedDataset:
    texts = dataset[cfg.task.text_field]
    labels = dataset[cfg.task.label_field]
    if limit is not None and limit < len(texts):
        texts = texts[:limit]
        labels = labels[:limit]
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=cfg.data.max_length,
        return_tensors="pt",
    )
    encodings = {k: v for k, v in enc.items()}  # input_ids, attention_mask, (token_type_ids)
    return TokenizedDataset(encodings, torch.tensor(labels, dtype=torch.long))


def load_dataloaders(
    cfg: Config,
    tokenizer,
    include_train: bool = True,
) -> Tuple[Optional[DataLoader], DataLoader]:
    """Return (train_loader | None, eval_loader) tokenized with ``tokenizer``."""
    from datasets import load_dataset

    name = cfg.task.dataset
    if cfg.task.subset:
        raw = load_dataset(name, cfg.task.subset)
    else:
        raw = load_dataset(name)

    eval_ds = _tokenize_split(raw[cfg.task.eval_split], tokenizer, cfg, cfg.data.max_eval_samples)
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg.train.eval_batch_size, shuffle=False, collate_fn=_collate
    )

    train_loader = None
    if include_train:
        train_ds = _tokenize_split(
            raw[cfg.task.train_split], tokenizer, cfg, cfg.data.max_train_samples
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=_collate
        )
    return train_loader, eval_loader


def sample_texts(cfg: Config, n: int = 32) -> List[str]:
    """Grab a handful of real eval texts for latency benchmarking / API smoke tests."""
    from datasets import load_dataset

    if cfg.task.subset:
        raw = load_dataset(cfg.task.dataset, cfg.task.subset, split=cfg.task.eval_split)
    else:
        raw = load_dataset(cfg.task.dataset, split=cfg.task.eval_split)
    texts = raw[cfg.task.text_field][:n]
    return list(texts)
