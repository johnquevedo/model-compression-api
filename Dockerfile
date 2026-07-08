# CPU inference image for the compressed model + FastAPI service.
# Training is expected to run separately (laptop/GPU); this image serves artifacts.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MCAPI_MODEL_DIR=/app/artifacts/student-quantized \
    MCAPI_BACKEND=quantized

WORKDIR /app

# Install CPU-only PyTorch first to keep the image small, then the rest.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# Install the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

# Bring in config + trained artifacts (build with artifacts/ present, or mount at runtime).
COPY config.yaml ./
COPY artifacts ./artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "mcapi.api:app", "--host", "0.0.0.0", "--port", "8000"]
