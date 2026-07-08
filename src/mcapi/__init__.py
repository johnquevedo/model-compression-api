"""Model Compression & Inference API.

A compact, laptop-runnable pipeline that:
  * fine-tunes a small student model on a text-classification task,
  * distills knowledge from a larger teacher into the student,
  * quantizes the student to int8,
  * (optionally) exports to ONNX,
  * benchmarks accuracy / latency / throughput / memory / size,
  * serves the optimized model behind a FastAPI endpoint.
"""

__version__ = "0.1.0"
