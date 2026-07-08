"""Command-line entrypoint tying the whole pipeline together.

Subcommands:
  eval-teacher   evaluate the teacher on the eval split
  train-student  fine-tune the student on hard labels (distillation baseline)
  distill        distill the teacher into the student
  quantize       dynamically int8-quantize a saved model
  export-onnx    export a saved model to ONNX (+ int8) via Optimum  [optional]
  benchmark      evaluate + benchmark every available artifact, write results
  report         render the comparison table from saved results
  pipeline       run the full end-to-end pipeline

Global options let you override config values, e.g.:
  mcapi distill --epochs 2 --batch-size 16 --alpha 0.7
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

from .config import Config, apply_overrides, load_config
from .utils import ensure_dir, get_device, load_json, save_json, set_seed


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--device", default=None, help="Force device: cuda|mps|cpu")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)


def _resolve_config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    overrides: Dict[str, object] = {
        "train.epochs": getattr(args, "epochs", None),
        "train.batch_size": getattr(args, "batch_size", None),
        "train.learning_rate": getattr(args, "learning_rate", None),
        "data.max_train_samples": getattr(args, "max_train_samples", None),
        "data.max_eval_samples": getattr(args, "max_eval_samples", None),
        "distill.temperature": getattr(args, "temperature", None),
        "distill.alpha": getattr(args, "alpha", None),
    }
    return apply_overrides(cfg, overrides)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_eval_teacher(args) -> None:
    from .data import load_dataloaders
    from .evaluate import evaluate_model
    from .models import load_teacher

    cfg = _resolve_config(args)
    device = get_device(args.device)
    print(f"[eval-teacher] device={device} teacher={cfg.models.teacher}")
    teacher, tokenizer = load_teacher(cfg, device)
    _, eval_loader = load_dataloaders(cfg, tokenizer, include_train=False)
    metrics = evaluate_model(teacher, eval_loader, device)
    print(f"[eval-teacher] {metrics}")
    save_json(metrics, os.path.join(cfg.paths.results_dir, "teacher_eval.json"))


def cmd_train_student(args) -> None:
    from .data import load_dataloaders
    from .models import load_student
    from .train import fine_tune

    cfg = _resolve_config(args)
    set_seed(cfg.train.seed)
    device = get_device(args.device)
    print(f"[train-student] device={device} student={cfg.models.student}")
    student, tokenizer = load_student(cfg, device)
    train_loader, eval_loader = load_dataloaders(cfg, tokenizer, include_train=True)
    result = fine_tune(student, tokenizer, train_loader, eval_loader, cfg, device, cfg.paths.student_dir)
    print(f"[train-student] {result['best_accuracy']:.4f} acc -> {result['output_dir']}")
    save_json(result, os.path.join(cfg.paths.results_dir, "student_finetune.json"))


def cmd_distill(args) -> None:
    from .data import load_dataloaders
    from .distill import distill
    from .models import load_student, load_teacher

    cfg = _resolve_config(args)
    set_seed(cfg.train.seed)
    device = get_device(args.device)
    print(f"[distill] device={device} T={cfg.distill.temperature} alpha={cfg.distill.alpha}")
    teacher, _ = load_teacher(cfg, device)
    student, tokenizer = load_student(cfg, device)
    # Distillation uses the STUDENT tokenizer for both models' inputs; teacher and
    # student here share the WordPiece vocab (BERT / DistilBERT), so this is valid.
    train_loader, eval_loader = load_dataloaders(cfg, tokenizer, include_train=True)
    result = distill(student, tokenizer, teacher, train_loader, eval_loader, cfg, device, cfg.paths.distilled_dir)
    print(f"[distill] {result['best_accuracy']:.4f} acc -> {result['output_dir']}")
    save_json(result, os.path.join(cfg.paths.results_dir, "distill.json"))


def cmd_quantize(args) -> None:
    from .quantize import quantize_model

    cfg = _resolve_config(args)
    source = args.source or cfg.paths.distilled_dir
    if not os.path.exists(source):
        source = cfg.paths.student_dir
    print(f"[quantize] source={source} -> {cfg.paths.quantized_dir}")
    meta = quantize_model(source, cfg.paths.quantized_dir, cfg.quantize.dtype)
    print(f"[quantize] size {meta['source_size_mb']:.1f}MB -> {meta['size_mb']:.1f}MB")
    save_json(meta, os.path.join(cfg.paths.results_dir, "quantize.json"))


def cmd_export_onnx(args) -> None:
    from .onnx_export import export_onnx

    cfg = _resolve_config(args)
    source = args.source or cfg.paths.distilled_dir
    if not os.path.exists(source):
        source = cfg.paths.student_dir
    print(f"[export-onnx] source={source} -> {cfg.paths.onnx_dir}")
    try:
        meta = export_onnx(source, cfg.paths.onnx_dir, quantize=True)
        print(f"[export-onnx] done: {meta}")
        save_json(meta, os.path.join(cfg.paths.results_dir, "onnx.json"))
    except ImportError as exc:
        print(f"[export-onnx] SKIPPED: {exc}")


def cmd_benchmark(args) -> None:
    from .benchmark import benchmark_torch_model
    from .data import load_dataloaders
    from .evaluate import evaluate_model
    from .models import load_classifier, load_teacher
    from .quantize import load_quantized_model
    from .report import render_markdown_report

    cfg = _resolve_config(args)
    device = get_device(args.device)
    records: Dict[str, Dict[str, object]] = {}

    # Teacher tokenizer drives shared eval loader (BERT/DistilBERT share vocab).
    teacher, teacher_tok = load_teacher(cfg, device)
    _, eval_loader = load_dataloaders(cfg, teacher_tok, include_train=False)

    def needs_tt(model) -> bool:
        # DistilBERT has no token_type_ids; BERT does.
        return "distilbert" not in model.config.model_type.lower()

    if not args.skip_teacher:
        print("[benchmark] teacher ...")
        records["teacher"] = {
            "eval": evaluate_model(teacher, eval_loader, device),
            "benchmark": benchmark_torch_model(teacher, cfg, device, cfg.models.teacher, needs_tt(teacher)),
        }

    for key, path in [
        ("student-finetuned", cfg.paths.student_dir),
        ("student-distilled", cfg.paths.distilled_dir),
    ]:
        if os.path.exists(path):
            print(f"[benchmark] {key} ...")
            model, _ = load_classifier(path, device)
            records[key] = {
                "eval": evaluate_model(model, eval_loader, device),
                "benchmark": benchmark_torch_model(model, cfg, device, path, needs_tt(model)),
            }

    # Quantized model runs on CPU.
    if os.path.exists(cfg.paths.quantized_dir):
        print("[benchmark] student-quantized (cpu) ...")
        import torch

        qmodel = load_quantized_model(cfg.paths.quantized_dir, cfg.quantize.dtype)
        cpu = torch.device("cpu")
        qbench = benchmark_torch_model(qmodel, cfg, cpu, cfg.paths.quantized_dir, needs_tt(qmodel))
        qmeta_path = os.path.join(cfg.paths.quantized_dir, "quantization.json")
        if os.path.exists(qmeta_path):
            qmeta = load_json(qmeta_path)
            qbench["param_count"] = qmeta.get("param_count", qbench.get("param_count"))
        records["student-quantized"] = {
            "eval": evaluate_model(qmodel, eval_loader, cpu),
            "benchmark": qbench,
        }

    ensure_dir(cfg.paths.results_dir)
    save_json(records, os.path.join(cfg.paths.results_dir, "benchmark.json"))
    report = render_markdown_report(records)
    report_path = os.path.join(cfg.paths.results_dir, "results.md")
    with open(report_path, "w") as fh:
        fh.write(report)
    print("\n" + report)
    print(f"\n[benchmark] wrote {report_path}")


def cmd_report(args) -> None:
    from .report import render_markdown_report

    cfg = _resolve_config(args)
    records = load_json(os.path.join(cfg.paths.results_dir, "benchmark.json"))
    print(render_markdown_report(records))


def cmd_pipeline(args) -> None:
    print("=== [1/5] Evaluate teacher ===")
    cmd_eval_teacher(args)
    print("=== [2/5] Fine-tune student (baseline) ===")
    cmd_train_student(args)
    print("=== [3/5] Distill teacher -> student ===")
    cmd_distill(args)
    print("=== [4/5] Quantize distilled student ===")
    args.source = None
    cmd_quantize(args)
    if not args.skip_onnx:
        print("=== [4b/5] Export ONNX ===")
        cmd_export_onnx(args)
    print("=== [5/5] Benchmark + report ===")
    cmd_benchmark(args)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(prog="mcapi", description="Model Compression & Inference API")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, extra in [
        ("eval-teacher", cmd_eval_teacher, None),
        ("train-student", cmd_train_student, None),
        ("distill", cmd_distill, None),
        ("quantize", cmd_quantize, "source"),
        ("export-onnx", cmd_export_onnx, "source"),
        ("benchmark", cmd_benchmark, "benchmark"),
        ("report", cmd_report, None),
        ("pipeline", cmd_pipeline, "pipeline"),
    ]:
        p = sub.add_parser(name)
        _add_common(p)
        if extra == "source":
            p.add_argument("--source", default=None, help="Source model dir to compress")
        if extra == "benchmark":
            p.add_argument("--skip-teacher", action="store_true")
        if extra == "pipeline":
            p.add_argument("--source", default=None)
            p.add_argument("--skip-teacher", action="store_true")
            p.add_argument("--skip-onnx", action="store_true")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
