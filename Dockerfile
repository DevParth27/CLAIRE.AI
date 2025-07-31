# Multi-stage build for smaller final image
FROM python:3.11-slim as builder
# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Copy and install Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM python:3.11-slim
# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app
# Copy application code
COPY --chown=app:app . .
# Create necessary directories
RUN mkdir -p database
# Set default port
ENV PORT=8000
# Expose port
EXPOSE 8000
# Health check - update this to use the actual port
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1
# Use the Python wrapper to start the server
CMD ["python", "run_server.py"]