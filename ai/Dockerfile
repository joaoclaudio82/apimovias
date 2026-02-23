FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY config ./config
COPY moviasai ./moviasai
COPY __init__.py ./__init__.py
COPY proxy.py ./proxy.py

RUN mkdir -p /app/data

EXPOSE 8010
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8010"]
