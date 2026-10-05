FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.21
COPY backend/pyproject.toml backend/uv.lock ./
COPY backend/app ./app
RUN uv sync --frozen --no-dev --extra faiss --no-editable
RUN useradd --uid 10001 --create-home appuser && mkdir /data && chown appuser:appuser /data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)"]
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

FROM runtime AS adapters
USER root
COPY training/requirements.txt /tmp/training-requirements.txt
RUN uv pip install --python /app/.venv/bin/python -r /tmp/training-requirements.txt
USER appuser
