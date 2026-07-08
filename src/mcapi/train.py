"""Plain student fine-tuning (hard-label cross-entropy) — the distillation baseline."""

from __future__ import annotations

import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .config import Config
from .evaluate import evaluate_model
from .utils import ensure_dir


def _build_optimizer_and_schedule(model, cfg: Config, num_training_steps: int):
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    warmup_steps = int(cfg.train.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def fine_tune(
    model,
    tokenizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    output_dir: str,
) -> Dict[str, object]:
    """Standard fine-tuning loop. Saves the best model (by accuracy) to ``output_dir``."""
    num_training_steps = len(train_loader) * cfg.train.epochs
    optimizer, scheduler = _build_optimizer_and_schedule(model, cfg, num_training_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    history = []
    start = time.time()

    for epoch in range(cfg.train.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(**batch.model_inputs()).logits
            loss = loss_fn(logits, batch.labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"[ft] epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        metrics = evaluate_model(model, eval_loader, device)
        avg_loss = running / max(1, len(train_loader))
        print(f"[ft] epoch {epoch} done avg_loss={avg_loss:.4f} eval={metrics}")
        history.append({"epoch": epoch, "train_loss": avg_loss, **metrics})

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            ensure_dir(output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start
    return {
        "output_dir": output_dir,
        "best_accuracy": best_acc,
        "train_seconds": elapsed,
        "history": history,
    }
