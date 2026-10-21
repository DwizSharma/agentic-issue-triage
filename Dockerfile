FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/model-cache

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app

COPY requirements-production.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-production.txt

COPY app.py triage_engine.py ./
RUN mkdir -p /app/model-cache /app/repo-cache \
    && chown -R app:app /app

USER app
EXPOSE 8000

# Liveness probe for standalone `docker run` (Compose overrides with /health/ready).
HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=5 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4); \
         sys.exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
