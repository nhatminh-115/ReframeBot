FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# CPU-only torch first — avoids pulling the 2 GB CUDA wheel
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install runtime deps (torch already satisfied above)
RUN pip install --no-cache-dir -e .

COPY app.py .
COPY .env.example .env.example

EXPOSE 8000

CMD ["python", "app.py"]
