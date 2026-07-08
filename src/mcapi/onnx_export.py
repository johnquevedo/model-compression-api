"""Optional ONNX export + int8 ONNX quantization via 🤗 Optimum / ONNX Runtime.

This is realistic for BERT/DistilBERT classifiers and gives a portable, CPU-optimized
artifact. It is optional because it pulls in `optimum` + `onnxruntime`; the pipeline
skips it gracefully (with a clear message) if those aren't installed.
"""

from __future__ import annotations

from typing import Dict

from .utils import dir_size_mb, ensure_dir, save_json


def export_onnx(model_dir: str, output_dir: str, quantize: bool = True) -> Dict[str, object]:
    """Export a saved HF classifier to ONNX and optionally int8-quantize it.

    Returns metadata including artifact sizes. Raises ImportError with a helpful
    message if the optional ONNX stack is missing.
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "ONNX export needs `optimum[onnxruntime]` and `onnxruntime`. "
            "Install them or run the pipeline with --skip-onnx."
        ) from exc

    ensure_dir(output_dir)

    # export=True traces the PyTorch model into ONNX on load.
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    ort_model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(model_dir).save_pretrained(output_dir)

    meta = {
        "source_dir": model_dir,
        "output_dir": output_dir,
        "quantized": False,
        "fp32_size_mb": dir_size_mb(output_dir),
    }

    if quantize:
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            quantizer = ORTQuantizer.from_pretrained(output_dir)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
            meta["quantized"] = True
            meta["int8_size_mb"] = dir_size_mb(output_dir)
        except Exception as exc:  # pragma: no cover - env dependent
            meta["quantize_error"] = str(exc)

    save_json(meta, f"{output_dir}/onnx_export.json")
    return meta
