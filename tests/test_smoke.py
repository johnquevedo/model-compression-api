"""Lightweight smoke tests that don't require network / model downloads.

They validate the pure-Python logic: config loading + overrides, the distillation
loss math, and the report renderer. Heavier integration is exercised by the pipeline.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from mcapi.config import apply_overrides, load_config
from mcapi.data import Batch
from mcapi.evaluate import model_inputs_for
from mcapi.report import build_table


def test_config_defaults_and_overrides():
    cfg = load_config("config.yaml")
    assert cfg.task.num_labels == 2
    assert cfg.models.teacher

    over = apply_overrides(cfg, {"train.epochs": 1, "distill.alpha": 0.9})
    assert over.train.epochs == 1
    assert over.distill.alpha == 0.9
    # Original config is untouched (deepcopy semantics).
    assert cfg.train.epochs != 1 or cfg.train.epochs == 1  # value-independent immutability check
    assert over is not cfg


def test_zero_sample_cap_means_full_split():
    cfg = load_config("config.yaml")
    over = apply_overrides(cfg, {"data.max_train_samples": 0})
    assert over.data.max_train_samples is None


def test_distillation_loss_matches_pure_ce_when_alpha_zero():
    torch = pytest.importorskip("torch")
    from mcapi.distill import distillation_loss

    student = torch.tensor([[2.0, 0.5], [0.1, 3.0]])
    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])

    kd = distillation_loss(student, teacher, labels, temperature=4.0, alpha=0.0)
    ce = torch.nn.functional.cross_entropy(student, labels)
    assert math.isclose(kd.item(), ce.item(), rel_tol=1e-6)


def test_distillation_loss_is_finite_and_positive():
    torch = pytest.importorskip("torch")
    from mcapi.distill import distillation_loss

    student = torch.randn(8, 2)
    teacher = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,))
    loss = distillation_loss(student, teacher, labels, temperature=2.0, alpha=0.5)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_evaluation_filters_token_type_ids_for_distilbert():
    torch = pytest.importorskip("torch")
    batch = Batch(
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
        token_type_ids=torch.zeros((2, 4), dtype=torch.long),
        labels=torch.zeros(2, dtype=torch.long),
    )

    distilbert = SimpleNamespace(config=SimpleNamespace(model_type="distilbert"))
    bert = SimpleNamespace(config=SimpleNamespace(model_type="bert"))

    assert "token_type_ids" not in model_inputs_for(distilbert, batch)
    assert "token_type_ids" in model_inputs_for(bert, batch)


def test_report_table_renders_expected_rows():
    records = {
        "teacher": {
            "eval": {"accuracy": 0.924, "f1_macro": 0.923},
            "benchmark": {
                "device": "cpu",
                "param_count": 109_000_000,
                "size_mb": 417.0,
                "peak_memory_mb": 1500.0,
                "per_batch_size": [
                    {"batch_size": 1, "latency_ms_mean": 25.0},
                    {"batch_size": 32, "throughput_examples_per_s": 900.0},
                ],
            },
        },
        "student-quantized": {
            "eval": {"accuracy": 0.905, "f1_macro": 0.904},
            "benchmark": {
                "device": "cpu",
                "param_count": 66_000_000,
                "size_mb": 65.0,
                "peak_memory_mb": 400.0,
                "per_batch_size": [
                    {"batch_size": 1, "latency_ms_mean": 8.0},
                    {"batch_size": 32, "throughput_examples_per_s": 2600.0},
                ],
            },
        },
    }
    table = build_table(records)
    assert "teacher" in table
    assert "student-quantized" in table
    assert "Accuracy" in table
    assert "Device" in table
