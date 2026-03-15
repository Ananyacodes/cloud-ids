FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY . .

# Non-root user
RUN useradd -m -u 1000 ids && chown -R ids:ids /app
USER ids

ENV IDS_CONFIG=/app/configs/config.yaml
EXPOSE 8080

CMD ["uvicorn", "inference.api.app:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "4", "--log-level", "info"]
