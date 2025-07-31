FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install with memory optimizations
COPY requirements-render.txt requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements.txt \
    && pip cache purge \
    && rm -rf ~/.cache/pip

# Copy application code
COPY . .

# Remove unnecessary files
RUN rm -rf __pycache__ \
    && rm -rf .git \
    && rm -rf tests \
    && rm -rf *.md

# Expose port
EXPOSE $PORT

# Run with memory limits
CMD uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1 --max-requests 100