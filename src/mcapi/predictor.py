"""Backend-agnostic predictor used by the API and by benchmarks.

Supports three backends selected by name:
  * ``torch``     — a saved fp32 HF classifier
  * ``quantized`` — a dynamically-quantized int8 model (CPU)
  * ``onnx``      — an ONNX Runtime model exported via Optimum
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .utils import get_device


class Predictor:
    def __init__(
        self,
        model_dir: str,
        backend: str = "torch",
        device: Optional[str] = None,
        max_length: int = 128,
        quant_dtype: str = "qint8",
    ):
        self.model_dir = model_dir
        self.backend = backend
        self.max_length = max_length
        # Quantized/ONNX backends are CPU-only.
        self.device = torch.device("cpu") if backend in {"quantized", "onnx"} else get_device(device)
        self._load(quant_dtype)

    def _load(self, quant_dtype: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        if self.backend == "torch":
            from transformers import AutoModelForSequenceClassification

            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label
        elif self.backend == "quantized":
            from .quantize import load_quantized_model

            self.model = load_quantized_model(self.model_dir, quant_dtype)
            self.id2label = self.model.config.id2label
        elif self.backend == "onnx":
            from optimum.onnxruntime import ORTModelForSequenceClassification

            self.model = ORTModelForSequenceClassification.from_pretrained(self.model_dir)
            self.id2label = self.model.config.id2label
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, object]]:
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.backend == "torch":
            enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()

        results = []
        for i, pid in enumerate(pred_ids):
            label = self.id2label.get(pid, str(pid)) if isinstance(self.id2label, dict) else str(pid)
            results.append(
                {
                    "label": label,
                    "label_id": int(pid),
                    "score": float(probs[i, pid].item()),
                    "probabilities": [float(x) for x in probs[i].tolist()],
                }
            )
        return results
