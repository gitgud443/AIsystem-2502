FROM python:3.11-slim

# Install build dependencies (only needed if any package still compiles)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libevent-dev \
        libssl-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clean up build deps (smaller image)
RUN apt-get purge -y --auto-remove \
        build-essential \
        libevent-dev \
        libssl-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 5004

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5004"]