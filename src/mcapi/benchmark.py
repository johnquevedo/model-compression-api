"""Inference-cost benchmarking: latency, throughput, memory, and model size.

Given a callable that maps a batch of encoded inputs -> logits, we measure:
  * latency  — per-forward wall-clock (mean / p50 / p95) at several batch sizes
  * throughput — examples/second
  * memory   — peak process RSS delta (CPU) or peak CUDA allocation (GPU)
  * size     — on-disk artifact size + parameter count
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .config import Config
from .utils import dir_size_mb


def _percentile(values: List[float], pct: float) -> float:
    return float(np.percentile(np.asarray(values), pct))


def _make_dummy_batch(
    batch_size: int, seq_len: int, needs_token_type: bool, device: torch.device, vocab_size: int = 1000
):
    hi = max(2, min(vocab_size, 30000))
    input_ids = torch.randint(0, hi, (batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if needs_token_type:
        inputs["token_type_ids"] = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=device
        )
    return inputs


def _peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


@torch.no_grad()
def benchmark_forward(
    forward_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    cfg: Config,
    device: torch.device,
    needs_token_type: bool = False,
    vocab_size: int = 1000,
) -> Dict[str, object]:
    """Benchmark a forward function across the configured batch sizes."""
    results: Dict[str, object] = {"device": str(device), "per_batch_size": []}
    seq_len = cfg.benchmark.seq_length

    for bs in cfg.benchmark.batch_sizes:
        batch = _make_dummy_batch(bs, seq_len, needs_token_type, device, vocab_size)

        # Warmup (also triggers lazy CUDA/MPS kernel compilation).
        for _ in range(cfg.benchmark.warmup_iters):
            forward_fn(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        latencies_ms: List[float] = []
        for _ in range(cfg.benchmark.measure_iters):
            t0 = time.perf_counter()
            forward_fn(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(latencies_ms))
        throughput = bs / (mean_ms / 1000.0)
        results["per_batch_size"].append(
            {
                "batch_size": bs,
                "latency_ms_mean": mean_ms,
                "latency_ms_p50": _percentile(latencies_ms, 50),
                "latency_ms_p95": _percentile(latencies_ms, 95),
                "throughput_examples_per_s": throughput,
            }
        )

    results["peak_memory_mb"] = _peak_memory_mb(device)
    return results


def torch_forward_fn(model) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """Wrap a HF model into a forward function returning logits."""

    def _fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return model(**batch).logits

    return _fn


def benchmark_torch_model(
    model,
    cfg: Config,
    device: torch.device,
    model_dir: Optional[str] = None,
    needs_token_type: bool = False,
) -> Dict[str, object]:
    """Benchmark a PyTorch HF model and attach size + parameter metadata."""
    from .utils import count_parameters

    model.eval()
    vocab_size = int(getattr(model.config, "vocab_size", 1000))
    out = benchmark_forward(torch_forward_fn(model), cfg, device, needs_token_type, vocab_size)
    out["param_count"] = count_parameters(model)
    if model_dir:
        out["size_mb"] = dir_size_mb(model_dir)
    return out
