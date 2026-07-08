# Model Compression & Inference API

A compact, laptop-runnable pipeline that **distills** a large teacher model into a
small student, **quantizes** it, **benchmarks** the quality-vs-cost tradeoff, and
**serves** the optimized model behind a FastAPI endpoint — with a concurrent-request
load test to measure latency under load.

- **Task:** binary sentiment classification (GLUE **SST-2**)
- **Teacher:** [`textattack/bert-base-uncased-SST-2`](https://huggingface.co/textattack/bert-base-uncased-SST-2) — BERT-base, already fine-tuned (~110M params, ~92% acc)
- **Student:** [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) — DistilBERT (~66M params), fine-tuned and distilled
- **Stack:** Python · PyTorch · Hugging Face (Transformers/Datasets) · ONNX Runtime · FastAPI · Docker

> Why SST-2? It's tiny (67k train / 872 validation), a standard distillation benchmark,
> and the teacher is already trained — so the whole pipeline runs on a laptop CPU in
> minutes (faster on a free Colab/T4 GPU). Test labels are private, so we evaluate on
> the validation split, as is standard for GLUE.

---

## What it demonstrates (resume bullets)

- **Built a compression pipeline** that distills a larger model into a smaller one while
  tracking accuracy, latency, memory, throughput, and model size.
- **Compared baseline, distilled, and quantized models** to quantify the tradeoff
  between model quality and inference cost.
- **Deployed the optimized model behind a FastAPI endpoint** and benchmarked response
  latency (p50/p95/p99) under concurrent requests.

---

## Architecture

```
              ┌────────────────────┐
              │  Teacher (BERT)    │  textattack/bert-base-uncased-SST-2
              │  ~110M, ~92% acc   │
              └─────────┬──────────┘
                        │ soft targets (KL, T=4)
                        ▼
 SST-2 ──► preprocess ─► Student (DistilBERT ~66M)
                        │      ├─ fine-tune (hard-label CE)      → baseline
                        │      └─ distill  (KD loss)             → distilled
                        ▼
                   quantize (int8 dynamic)                       → quantized
                        │
                   (optional) ONNX export + int8                 → onnx
                        ▼
        benchmark: accuracy · latency · throughput · memory · size
                        ▼
                FastAPI /predict  ──►  concurrent load test
```

Pipeline stages (each maps to a module in `src/mcapi/`):

| Stage | Module | Description |
|-------|--------|-------------|
| Data loading & preprocessing | `data.py` | Load + tokenize SST-2 into DataLoaders |
| Teacher evaluation | `evaluate.py`, `models.py` | Baseline teacher accuracy/F1 |
| Student fine-tuning | `train.py` | Hard-label cross-entropy (distillation baseline) |
| Knowledge distillation | `distill.py` | `α·T²·KL(student‖teacher) + (1−α)·CE` |
| Quantization | `quantize.py` | Dynamic int8 quantization of Linear layers |
| ONNX export (optional) | `onnx_export.py` | Optimum → ONNX + int8 (skips cleanly if absent) |
| Metrics | `metrics.py` | Accuracy + macro-F1 |
| Benchmarking | `benchmark.py` | Latency / throughput / memory / size |
| API | `api.py`, `predictor.py` | FastAPI `/predict` with pluggable backend |
| Concurrent benchmark | `scripts/bench_concurrent.py` | Async load test, latency percentiles |
| Container | `Dockerfile`, `docker-compose.yml` | CPU inference image |

---

## Setup

Requires Python 3.9+ (3.11 recommended). A GPU is optional — CPU works for this task.

```bash
git clone <your-repo-url> model-compression-api
cd model-compression-api

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the `mcapi` CLI
```

> **CPU-only PyTorch** (smaller install):
> `pip install torch --index-url https://download.pytorch.org/whl/cpu` before `pip install -r requirements.txt`.

---

## Quickstart

Run the whole thing end to end (small settings for a laptop):

```bash
mcapi pipeline --epochs 2 --max-train-samples 8000
# or: make pipeline
```

This runs: evaluate teacher → fine-tune student → distill → quantize → (ONNX) →
benchmark, writing artifacts to `artifacts/` and a results table to
`artifacts/results/results.md`.

### Or run each stage individually

```bash
mcapi eval-teacher                       # teacher baseline accuracy/F1
mcapi train-student --epochs 3           # fine-tune student (baseline)
mcapi distill --epochs 3 --alpha 0.5 --temperature 4.0   # knowledge distillation
mcapi quantize                           # int8-quantize the distilled student
mcapi export-onnx                        # optional ONNX export (skips if optimum missing)
mcapi benchmark                          # evaluate + benchmark all artifacts
mcapi report                             # re-print the comparison table
```

Common overrides (any stage): `--epochs`, `--batch-size`, `--learning-rate`,
`--max-train-samples`, `--max-eval-samples`, `--temperature`, `--alpha`, `--device cpu|cuda|mps`.

---

## Serving the model

```bash
# Serve the quantized model (default). Set MCAPI_BACKEND=torch or onnx to switch.
export MCAPI_MODEL_DIR=artifacts/student-quantized
export MCAPI_BACKEND=quantized
uvicorn mcapi.api:app --host 0.0.0.0 --port 8000
# or: make serve
```

Endpoints:

- `GET /healthz` — liveness + whether the model is loaded
- `GET /info` — backend, device, labels, model dir
- `POST /predict` — classify one or more texts

```bash
curl -s http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"texts": ["this movie was fantastic", "what a waste of two hours"]}' | python -m json.tool
```

```json
{
  "predictions": [
    {"label": "LABEL_1", "label_id": 1, "score": 0.998, "probabilities": [0.002, 0.998]},
    {"label": "LABEL_0", "label_id": 0, "score": 0.994, "probabilities": [0.994, 0.006]}
  ],
  "latency_ms": 7.8,
  "backend": "quantized",
  "batch_size": 2
}
```

### Concurrent request benchmark

With the server running:

```bash
python scripts/bench_concurrent.py --url http://localhost:8000/predict \
    --requests 500 --concurrency 20 --batch-size 1
# or: make loadtest
```

Reports throughput (req/s) and latency **p50 / p90 / p95 / p99** under load.

---

## Docker

The image is CPU-only and serves whatever is in `artifacts/`. Train first (or mount a
prepared `artifacts/`), then:

```bash
make docker-build
make docker-run          # -> http://localhost:8000
# or with compose (mounts ./artifacts read-only):
docker compose up --build
```

---

## Results

Generated by `mcapi benchmark --device cpu` → `artifacts/results/results.md`.

| Model             | Device   |   Accuracy |   F1 (macro) |   Params (M) |   Size (MB) |   Latency bs=1 (ms) |   Throughput bs=32 (ex/s) |   Peak mem (MB) |
|-------------------|----------|------------|--------------|--------------|-------------|---------------------|---------------------------|-----------------|
| teacher           | cpu      |     0.9243 |       0.9243 |        109.5 |       417.7 |               28.29 |                      53.3 |          1388.8 |
| student-finetuned | cpu      |     0.8865 |       0.8864 |         67   |       256.3 |               14.43 |                     106.1 |          1676.9 |
| student-distilled | cpu      |     0.8819 |       0.8817 |         67   |       256.3 |               14.54 |                     103.6 |          1678   |
| student-quantized | cpu      |     0.8807 |       0.8807 |         67   |       133.2 |               32.41 |                      31.1 |          1460.4 |

**Concurrent load test** (`scripts/bench_concurrent.py`, quantized backend):

| Backend | Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) |
|---------|------------:|-------------------:|---------:|---------:|---------:|
| quantized | 20 | 230.7 | 85.6 | 95.5 | 109.3 |

**How to read it:** the DistilBERT student uses ~61% of the teacher's parameters and
roughly doubles CPU throughput in this run, while retaining most of the teacher's
accuracy. Dynamic int8 quantization cuts the saved artifact from 256.3 MB to 133.2 MB
with a small accuracy drop, though this PyTorch/QNNPACK run is slower on this Mac CPU.

---

## Design notes

- **Distillation loss** (`distill.py`): `α·T²·KL(softmax(student/T) ‖ softmax(teacher/T)) + (1−α)·CE(student, labels)`.
  The `T²` factor keeps soft-target gradients on the same scale as the hard-label term
  (Hinton et al., 2015). Teacher logits are computed under `no_grad`.
- **Shared vocab:** BERT and DistilBERT share the WordPiece vocabulary, so the student
  tokenizer feeds both models during distillation.
- **Quantization** is dynamic int8 on `nn.Linear` (no calibration set needed) and targets
  CPU deployment. Quantized weights are saved as a state dict + config/tokenizer, and
  reconstructed by `quantize.load_quantized_model`.
- **Benchmark methodology:** warmup iters then N timed forwards on synthetic batches at
  several batch sizes; latency reported as mean/p50/p95, throughput as examples/s, memory
  as peak CUDA allocation (GPU) or process RSS (CPU). See `benchmark.py`.
- **ONNX is optional** by design — it needs `optimum[onnxruntime]`; the pipeline logs a
  clear skip message if unavailable, so the core path never breaks.

---

## Project layout

```
model-compression-api/
├── config.yaml               # all defaults (overridable on the CLI)
├── requirements.txt
├── pyproject.toml            # installs the `mcapi` CLI
├── Dockerfile / docker-compose.yml
├── Makefile
├── src/mcapi/
│   ├── cli.py                # subcommands + full pipeline
│   ├── config.py             # typed config + overrides
│   ├── data.py               # dataset load + tokenize -> DataLoaders
│   ├── models.py             # teacher/student/classifier loaders
│   ├── train.py              # student fine-tuning
│   ├── distill.py            # knowledge distillation
│   ├── quantize.py           # dynamic int8 quantization
│   ├── onnx_export.py        # optional ONNX export
│   ├── evaluate.py           # accuracy/F1 evaluation
│   ├── metrics.py            # metric computation
│   ├── benchmark.py          # latency/throughput/memory/size
│   ├── report.py             # comparison table renderer
│   ├── predictor.py          # backend-agnostic inference
│   └── api.py                # FastAPI service
├── scripts/bench_concurrent.py   # async concurrent load test
├── tests/test_smoke.py
└── artifacts/                # trained models + results (gitignored)
```

## Testing

```bash
pytest -q          # smoke tests: config, distillation-loss math, report renderer
```

## License

MIT
