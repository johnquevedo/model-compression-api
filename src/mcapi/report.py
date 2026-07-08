"""Render the baseline vs distilled vs quantized comparison table.

Consumes the per-model result records produced by the benchmark command and emits a
Markdown table capturing the quality-vs-cost tradeoff.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from tabulate import tabulate

COLUMNS = [
    "Model",
    "Device",
    "Accuracy",
    "F1 (macro)",
    "Params (M)",
    "Size (MB)",
    "Latency bs=1 (ms)",
    "Throughput bs=32 (ex/s)",
    "Peak mem (MB)",
]


def _row_for(name: str, rec: Dict[str, object]) -> List[object]:
    eval_m = rec.get("eval", {}) or {}
    bench = rec.get("benchmark", {}) or {}
    per_bs = {b["batch_size"]: b for b in bench.get("per_batch_size", [])}

    def fmt(v, nd=4):
        return round(v, nd) if isinstance(v, (int, float)) else "-"

    params = bench.get("param_count")
    lat1 = per_bs.get(1, {}).get("latency_ms_mean")
    tp32 = per_bs.get(32, {}).get("throughput_examples_per_s")
    # Fall back to the largest measured batch size if 32 wasn't benchmarked.
    if tp32 is None and per_bs:
        largest = per_bs[max(per_bs)]
        tp32 = largest.get("throughput_examples_per_s")

    return [
        name,
        bench.get("device", "-"),
        fmt(eval_m.get("accuracy")),
        fmt(eval_m.get("f1_macro")),
        fmt(params / 1e6, 1) if isinstance(params, (int, float)) else "-",
        fmt(bench.get("size_mb"), 1),
        fmt(lat1, 2),
        fmt(tp32, 1),
        fmt(bench.get("peak_memory_mb"), 1),
    ]


def build_table(records: Dict[str, Dict[str, object]], fmt: str = "github") -> str:
    order = ["teacher", "student-finetuned", "student-distilled", "student-quantized", "student-onnx"]
    rows = []
    for key in order:
        if key in records:
            rows.append(_row_for(key, records[key]))
    for key, rec in records.items():
        if key not in order:
            rows.append(_row_for(key, rec))
    return tabulate(rows, headers=COLUMNS, tablefmt=fmt)


def render_markdown_report(records: Dict[str, Dict[str, object]]) -> str:
    table = build_table(records, fmt="github")
    lines = [
        "# Model Compression Results",
        "",
        "Comparison of baseline, distilled, and quantized models "
        "(quality vs. inference cost).",
        "",
        table,
        "",
        "_Latency/throughput measured with synthetic batches at seq_len from config; "
        "accuracy on the eval split. Device is shown because quantized PyTorch "
        "models run on CPU. Absolute numbers depend on your hardware._",
        "",
    ]
    return "\n".join(lines)
