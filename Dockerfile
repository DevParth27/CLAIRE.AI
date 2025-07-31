FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-render.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && rm -rf ~/.cache/pip

# Copy application code
COPY . .

# Remove unnecessary files to reduce image size
RUN rm -rf __pycache__ .git .gitignore README.md requirements.txt requirements-vercel.txt

# Expose port (Render will set PORT env var)
EXPOSE $PORT

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1 --limit-max-requests 100