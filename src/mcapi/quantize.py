"""Post-training dynamic int8 quantization of a saved student model (CPU inference).

Dynamic quantization converts the ``nn.Linear`` weights to int8 and quantizes
activations on the fly. It needs no calibration data, works well for
transformer/BERT-style models, and is CPU-only — which matches a cheap deployment.
"""

from __future__ import annotations

from typing import Dict

import torch

from .utils import count_parameters, dir_size_mb, ensure_dir, save_json

_DTYPES = {"qint8": torch.qint8, "quint8": torch.quint8, "float16": torch.float16}


def _select_qengine() -> str:
    """Pick a supported quantization backend: FBGEMM (x86) if present, else QNNPACK (ARM)."""
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    engine = "fbgemm" if "fbgemm" in supported else (supported[0] if supported else None)
    if engine is None:
        raise RuntimeError(
            "No quantization engine available in this PyTorch build "
            "(need fbgemm or qnnpack). Try a standard CPU build of torch."
        )
    torch.backends.quantized.engine = engine
    return engine


def quantize_model(model_dir: str, output_dir: str, dtype: str = "qint8") -> Dict[str, object]:
    """Load a saved HF classifier, dynamically quantize it, and save the state dict.

    Quantized models are saved as a ``pytorch_model_quantized.pt`` state dict plus the
    original config/tokenizer, because HF ``save_pretrained`` cannot serialize the
    quantized modules directly. Load it back with :func:`load_quantized_model`.
    """
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    torch_dtype = _DTYPES.get(dtype, torch.qint8)
    engine = _select_qengine()

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to("cpu")

    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch_dtype
    )

    ensure_dir(output_dir)
    # Persist config + tokenizer so the directory is self-describing.
    AutoConfig.from_pretrained(model_dir).save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(model_dir).save_pretrained(output_dir)
    state_path = f"{output_dir}/pytorch_model_quantized.pt"
    torch.save(quantized.state_dict(), state_path)

    meta = {
        "source_dir": model_dir,
        "output_dir": output_dir,
        "dtype": dtype,
        "engine": engine,
        "quantized_state_dict": "pytorch_model_quantized.pt",
        "size_mb": dir_size_mb(output_dir),
        "source_size_mb": dir_size_mb(model_dir),
        "param_count": count_parameters(model),
    }
    save_json(meta, f"{output_dir}/quantization.json")
    return meta


def load_quantized_model(model_dir: str, dtype: str = "qint8"):
    """Reconstruct a dynamically-quantized classifier from a quantized artifact dir."""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    torch_dtype = _DTYPES.get(dtype, torch.qint8)
    _select_qengine()
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_config(config)
    model.eval()
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch_dtype)
    state = torch.load(f"{model_dir}/pytorch_model_quantized.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
