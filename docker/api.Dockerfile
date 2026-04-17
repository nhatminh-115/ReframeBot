FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# Install runtime deps only (no torch/transformers/peft — handled by vLLM container)
RUN pip install --no-cache-dir -e .

COPY app.py .
COPY .env.example .env.example

EXPOSE 8000

CMD ["python", "app.py"]
