.PHONY: help install eval-teacher train distill quantize onnx benchmark pipeline serve loadtest test docker-build docker-run clean

PY ?= python
PORT ?= 8000

help:
	@echo "Targets:"
	@echo "  install       Install package + dependencies"
	@echo "  pipeline      Run full pipeline (eval->train->distill->quantize->benchmark)"
	@echo "  eval-teacher  Evaluate teacher on eval split"
	@echo "  train         Fine-tune student (baseline)"
	@echo "  distill       Distill teacher -> student"
	@echo "  quantize      Int8-quantize distilled student"
	@echo "  onnx          Export distilled student to ONNX"
	@echo "  benchmark     Evaluate + benchmark all artifacts, write results table"
	@echo "  serve         Run FastAPI server (uvicorn) on PORT=$(PORT)"
	@echo "  loadtest      Concurrent request benchmark against the server"
	@echo "  test          Run smoke tests"
	@echo "  docker-build  Build the CPU inference image"
	@echo "  docker-run    Run the container on PORT=$(PORT)"

install:
	pip install -r requirements.txt
	pip install -e .

eval-teacher:
	mcapi eval-teacher

train:
	mcapi train-student

distill:
	mcapi distill

quantize:
	mcapi quantize

onnx:
	mcapi export-onnx

benchmark:
	mcapi benchmark

# Small, laptop-friendly end-to-end run.
pipeline:
	mcapi pipeline --epochs 2 --max-train-samples 8000

serve:
	uvicorn mcapi.api:app --host 0.0.0.0 --port $(PORT)

loadtest:
	$(PY) scripts/bench_concurrent.py --url http://localhost:$(PORT)/predict --requests 500 --concurrency 20

test:
	$(PY) -m pytest -q

docker-build:
	docker build -t model-compression-api .

docker-run:
	docker run --rm -p $(PORT):8000 -v $(PWD)/artifacts:/app/artifacts:ro model-compression-api

clean:
	rm -rf artifacts/student-* artifacts/results
	find . -type d -name __pycache__ -exec rm -rf {} +
