# Base Image
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build Stage
FROM base AS builder
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
  && pip install --no-cache-dir -r requirements.txt

# Runtime Stage
FROM base AS runtime
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY app.py ./app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
