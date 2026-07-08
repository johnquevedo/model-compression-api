"""Knowledge distillation: transfer a large teacher's soft targets into the student.

Loss = alpha * T^2 * KL(softmax(student/T) || softmax(teacher/T))
       + (1 - alpha) * CrossEntropy(student_logits, hard_labels)

The T^2 factor keeps the soft-target gradient magnitude comparable to the hard-label
term (Hinton et al., 2015). Teacher logits are computed on the fly with no_grad.
"""

from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .evaluate import evaluate_model
from .train import _build_optimizer_and_schedule
from .utils import ensure_dir


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    t = temperature
    soft_teacher = F.softmax(teacher_logits / t, dim=-1)
    soft_student = F.log_softmax(student_logits / t, dim=-1)
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (t * t)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1.0 - alpha) * ce


def distill(
    student,
    student_tokenizer,
    teacher,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    output_dir: str,
) -> Dict[str, object]:
    """Distill ``teacher`` into ``student``. Saves best student (by accuracy)."""
    num_training_steps = len(train_loader) * cfg.train.epochs
    optimizer, scheduler = _build_optimizer_and_schedule(student, cfg, num_training_steps)

    teacher.eval()
    best_acc = -1.0
    history = []
    start = time.time()

    for epoch in range(cfg.train.epochs):
        student.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            with torch.no_grad():
                teacher_logits = teacher(**batch.model_inputs()).logits
            optimizer.zero_grad()
            student_logits = student(**batch.model_inputs()).logits
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                batch.labels,
                cfg.distill.temperature,
                cfg.distill.alpha,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"[kd] epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        metrics = evaluate_model(student, eval_loader, device)
        avg_loss = running / max(1, len(train_loader))
        print(f"[kd] epoch {epoch} done avg_loss={avg_loss:.4f} eval={metrics}")
        history.append({"epoch": epoch, "train_loss": avg_loss, **metrics})

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            ensure_dir(output_dir)
            student.save_pretrained(output_dir)
            student_tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start
    return {
        "output_dir": output_dir,
        "best_accuracy": best_acc,
        "train_seconds": elapsed,
        "temperature": cfg.distill.temperature,
        "alpha": cfg.distill.alpha,
        "history": history,
    }
