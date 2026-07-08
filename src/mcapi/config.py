"""Typed configuration loaded from ``config.yaml`` with CLI overrides."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TaskConfig:
    dataset: str = "glue"
    subset: Optional[str] = "sst2"
    text_field: str = "sentence"
    label_field: str = "label"
    num_labels: int = 2
    train_split: str = "train"
    eval_split: str = "validation"


@dataclass
class ModelsConfig:
    teacher: str = "textattack/bert-base-uncased-SST-2"
    student: str = "distilbert-base-uncased"


@dataclass
class DataConfig:
    max_length: int = 128
    max_train_samples: Optional[int] = 20000
    max_eval_samples: Optional[int] = 872


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42


@dataclass
class DistillConfig:
    temperature: float = 4.0
    alpha: float = 0.5


@dataclass
class QuantizeConfig:
    dtype: str = "qint8"


@dataclass
class BenchmarkConfig:
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])
    warmup_iters: int = 5
    measure_iters: int = 50
    seq_length: int = 128


@dataclass
class PathsConfig:
    artifacts_dir: str = "artifacts"
    student_dir: str = "artifacts/student-finetuned"
    distilled_dir: str = "artifacts/student-distilled"
    quantized_dir: str = "artifacts/student-quantized"
    onnx_dir: str = "artifacts/student-onnx"
    results_dir: str = "artifacts/results"


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _build(cls, data: Optional[Dict[str, Any]]):
    """Recursively instantiate a (possibly nested) dataclass from a dict."""
    if data is None:
        return cls()
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _build(f.type, value)
        else:
            # Normalise a few YAML quirks (0 / null -> None for "use everything").
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: Optional[str] = None) -> Config:
    """Load config from YAML. Falls back to dataclass defaults if the file is absent."""
    raw: Dict[str, Any] = {}
    path = path or os.environ.get("MCAPI_CONFIG", "config.yaml")
    if path and os.path.exists(path):
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
    cfg = Config(
        task=_build(TaskConfig, raw.get("task")),
        models=_build(ModelsConfig, raw.get("models")),
        data=_build(DataConfig, raw.get("data")),
        train=_build(TrainConfig, raw.get("train")),
        distill=_build(DistillConfig, raw.get("distill")),
        quantize=_build(QuantizeConfig, raw.get("quantize")),
        benchmark=_build(BenchmarkConfig, raw.get("benchmark")),
        paths=_build(PathsConfig, raw.get("paths")),
    )
    return _normalise(cfg)


def _normalise(cfg: Config) -> Config:
    # Treat 0 or negative sample caps as "use the whole split".
    if cfg.data.max_train_samples is not None and cfg.data.max_train_samples <= 0:
        cfg.data.max_train_samples = None
    if cfg.data.max_eval_samples is not None and cfg.data.max_eval_samples <= 0:
        cfg.data.max_eval_samples = None
    return cfg


def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Apply flat CLI overrides like ``{"train.epochs": 2}`` onto a config copy."""
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        if value is None:
            continue
        section, _, attr = dotted.partition(".")
        target = getattr(cfg, section)
        setattr(target, attr, value)
    return _normalise(cfg)
